import asyncio
import logging
import re
import secrets
import time

from telethon import Button, errors, types

from .analysis import extract, index_groups, prepare_for_scoring, report
from .encoder import Encoder, ensure_model
from .telegram import RetryLater, channel_reference
from .worker import Route, deliver, index_batch

LOG = logging.getLogger(__name__)
POST_LINK = re.compile(r'(?:https?://)?t\.me/(?:s/)?([A-Za-z][A-Za-z0-9_]{3,31})/(\d+)(?:\?.*)?')
PRIVATE_LINK = re.compile(r'(?:https?://)?t\.me/c/(\d+)/(\d+)(?:\?.*)?')
HELP = (
    'Бот собирает историю канала в базу. Публикация новых постов включается отдельно.\n\n'
    '/channel @источник @мой_канал — выбрать каналы\n'
    '/post on|off — копировать новые посты и отвечать оценкой в моём канале\n'
    '/index on|off — собирать базу в фоне\n'
    '/llm on|off — включить или выключить MiniLM\n'
    '/an — проверить ссылку, пересланный пост или текст в личке\n'
    '/status — режимы, прогресс, очередь и ошибки\n'
    '/queue — очередь; /queue clear|pause|resume — управление\n'
    '/stop — остановить публикацию, индексацию и очередь'
)


class Control:
    def __init__(self, archive, telegram, model_dir, admins):
        self.archive = archive
        self.telegram = telegram
        self.model_dir = model_dir
        self.admins = admins
        self.mode = archive.setting('mode', 'classic')
        self.encoder = None
        self.model_state = 'выключена' if self.mode == 'classic' else 'загрузка'
        self.model_error = None
        self.routes = {}
        self.count_cache = {}
        self.waiting = set()
        self.index_offers = {}
        self.last_offer = {}
        self.command_lock = asyncio.Lock()
        self.delivery_locks = {}
        self.generation = 0
        for source, peers, _ in archive.route_records():
            self.routes[source] = Route(*(types.InputPeerChannel(*peer) for peer in peers[:2]))

    async def warm_model(self):
        if self.mode == 'minilm':
            try:
                await self.set_llm(True)
            except Exception:
                for source in self.routes:
                    self.archive.set_auto_publish(source, 0)
                    self.archive.set_delivery_enabled(source, 0)
                raise

    async def set_llm(self, enabled):
        if enabled and self.encoder is None:
            previous_mode = self.mode
            self.mode = 'minilm'
            self.model_state = 'загрузка'
            try:
                await asyncio.to_thread(ensure_model, self.model_dir)
                encoder = await asyncio.to_thread(Encoder, self.model_dir)
                self.archive.set_encoder(encoder)
                self.encoder = encoder
                self.model_state = 'готова'
                self.model_error = None
            except Exception as exc:
                self.mode = previous_mode
                self.model_state = 'ошибка'
                self.model_error = str(exc)
                raise
        if not enabled:
            self.encoder = None
            self.archive.set_encoder(None)
            self.model_state = 'выключена'
        self.mode = 'minilm' if enabled else 'classic'
        self.archive.set_setting('mode', self.mode)

    async def channel(self, source_ref, destination_ref):
        route = await self.telegram.configure(source_ref, destination_ref)
        source = route.source.channel_id
        destination = route.destination.channel_id
        head = await self.telegram.head(route.source)
        count = await self.telegram.count(route.source)
        old = self.archive.state(source)
        peers = [(peer.channel_id, peer.access_hash) for peer in (route.source, route.destination)]
        self.archive.save_route(source, destination, peers, active=old['active'] if old else 1)
        if old is None:
            self.archive.set_route_active(source, 1)
            self.archive.set_auto_publish(source, 0)
            self.archive.set_delivery_enabled(source, 0)
            self.archive.set_posting_cursor(source, head)
        self.archive.set_setting(f'route_name:{source}', f'{source_ref} → {destination_ref}')
        self.routes[source] = route
        self.count_cache[source] = (count, time.monotonic())
        self.generation += 1
        return (f'✅ {source_ref} → {destination_ref}. В источнике около {count} сообщений. '
                f'Индексация {"включена" if self.archive.route_active(source) else "выключена"}, '
                f'публикация {"включена" if self.archive.state(source)["auto_publish"] else "выключена"}. '
                'Прогресс: /status.')

    async def source_id(self, reference):
        ref = channel_reference(reference)
        if isinstance(ref, int):
            return int(str(ref)[4:])
        peer = await self.telegram.call('user', self.telegram.user.get_input_entity, ref)
        if not isinstance(peer, types.InputPeerChannel):
            raise ValueError('Нужна ссылка на канал')
        return peer.channel_id

    def score_text(self, source, text, before=2**63 - 1):
        if not text or not text.strip():
            return 'У поста нет текста для оценки.'
        features = extract(text, self.encoder)
        probes = list(index_groups(features, query=True))
        scores = self.archive.score(source, before, prepare_for_scoring(features), probes)
        return report(scores)

    def offer_index(self, sender, source, post, text):
        if sender is None or not text or not text.strip():
            return
        token = secrets.token_urlsafe(12)
        self.index_offers[token] = (sender, source, post, text)
        self.last_offer[sender] = token
        if len(self.index_offers) > 100:
            self.index_offers.pop(next(iter(self.index_offers)))

    async def analyze(self, value=None, forward=None, sender=None):
        if not self.routes:
            raise ValueError('Сначала выбери каналы: /channel @источник @мой_канал')
        if forward and isinstance(forward.from_id, types.PeerChannel) and forward.channel_post:
            source = forward.from_id.channel_id
            if source not in self.routes:
                raise ValueError('Пересланный пост не из настроенного канала')
            post = forward.channel_post
            text = await self.telegram.call('user', self.telegram.user.get_messages,
                                            self.routes[source].source, ids=post)
            if text is None:
                raise ValueError('Пост не найден')
            self.offer_index(sender, source, post, text.message)
            return f'Пост {post}, база {self.archive.indexed_count(source)}:\n' + self.score_text(
                source, text.message, post)
        value = (value or '').strip()
        public = POST_LINK.fullmatch(value)
        private = PRIVATE_LINK.fullmatch(value)
        if public or private:
            match = public or private
            reference = '@' + public[1] if public else '-100' + private[1]
            source = await self.source_id(reference)
            if source not in self.routes:
                raise ValueError('Этот канал не настроен')
            post = int(match[2])
            item = await self.telegram.call('user', self.telegram.user.get_messages,
                                            self.routes[source].source, ids=post)
            if item is None:
                raise ValueError('Пост не найден')
            self.offer_index(sender, source, post, item.message)
            return f'Пост {post}, база {self.archive.indexed_count(source)}:\n' + self.score_text(
                source, item.message, post)
        if value.startswith(('t.me/', 'https://t.me/', 'http://t.me/')):
            raise ValueError('Пришли ссылку на отдельный пост, например https://t.me/channel/123')
        if not value:
            return None
        source = None
        if value.startswith('@') and ' ' in value:
            reference, value = value.split(' ', 1)
            source = await self.source_id(reference)
        elif len(self.routes) == 1:
            source = next(iter(self.routes))
        if source not in self.routes:
            raise ValueError('Для текста укажи канал: /an @канал текст')
        self.offer_index(sender, source, None, value)
        return (f'Сравнение с базой {self.archive.indexed_count(source)} постов:\n'
                + self.score_text(source, value))

    async def index(self, source):
        if self.mode == 'minilm' and self.encoder is None:
            return False
        if source not in self.routes or not self.archive.route_active(source):
            return False
        generation = self.generation
        return await index_batch(self.archive, self.telegram, self.routes[source].source,
                                 should_continue=lambda: generation == self.generation
                                 and self.archive.route_active(source))

    async def watch(self, source):
        if self.mode == 'minilm' and self.encoder is None:
            return False
        if source not in self.routes or not self.archive.state(source)['auto_publish']:
            return False
        route = self.routes[source]
        cursor = self.archive.state(source)['posting_cursor']
        head = await self.telegram.head(route.source)
        if head <= cursor:
            return False
        count = 0
        async for item in self.telegram.history(route.source, cursor, head, 20):
            if not self.archive.state(source)['auto_publish']:
                return False
            result = self.score_text(source, item.message, item.id)
            self.archive.enqueue(source, item.id, result)
            self.archive.set_posting_cursor(source, item.id)
            count += 1
        return count == 20

    async def deliver(self, source):
        if self.mode == 'minilm' and self.encoder is None:
            return False
        state = self.archive.state(source)
        if not state or not state['auto_publish'] or not self.archive.delivery_enabled(source):
            return False
        lock = self.delivery_locks.setdefault(source, asyncio.Lock())
        async with lock:
            return await deliver(self.archive, self.telegram, self.routes[source],
                                 should_continue=lambda: self.archive.state(source)['auto_publish']
                                 and self.archive.delivery_enabled(source))

    async def status(self):
        lines = [f'📤 Публикация: {"вкл" if any(self.archive.state(s)["auto_publish"] for s in self.routes) else "выкл"}',
                 f'📚 Индексация: {"вкл" if any(self.archive.route_active(s) for s in self.routes) else "выкл"}',
                 f'🧠 MiniLM: {"вкл" if self.mode == "minilm" and self.encoder else "выкл"} ({self.model_state})']
        if self.model_error:
            lines.append(f'⚠️ Модель: {self.model_error[:250]}')
        for source, route in self.routes.items():
            row = self.archive.state(source)
            try:
                cached, checked = self.count_cache.get(source, (None, 0))
                if time.monotonic() - checked > 120:
                    cached = await self.telegram.count(route.source)
                    self.count_cache[source] = (cached, time.monotonic())
            except (OSError, errors.RPCError, RetryLater):
                cached = self.count_cache.get(source, (None, 0))[0]
            indexed = self.archive.indexed_count(source)
            new = max(0, cached - row['scan_count']) if cached is not None else '?'
            name = self.archive.setting(f'route_name:{source}', f'-100{source}')
            lines.append(f'\n{name}\n📨 в канале: {cached if cached is not None else "?"}; '
                         f'просмотрено: {row["scan_count"]}; в базе: {indexed}; осталось просмотреть: ~{new}\n'
                         f'⏳ очередь: {self.archive.queue_size(source)} '
                         f'({"идёт" if self.archive.delivery_enabled(source) and row["auto_publish"] else "пауза"})')
            if error := self.archive.error(source):
                lines.append(f'⚠️ {error[:250]}')
        if not self.routes:
            lines.append('Каналы ещё не настроены. Начни с /channel @источник @мой_канал')
        return '\n'.join(lines)

    async def dispatch(self, text, sender=None):
        command, _, argument = text.strip().partition(' ')
        command = command.lower().split('@', 1)[0]
        argument = argument.strip()
        if command != '/an':
            self.waiting.discard(sender)
        if command == '/stop':
            self.generation += 1
            self.archive.stop_all()
            return '🛑 Публикация, индексация и очередь остановлены.'
        if command in ('/start', '/help'):
            return HELP
        if command == '/status':
            return await self.status()
        if command == '/queue':
            if argument == 'clear':
                return f'🧹 Из очереди убрано: {self.archive.clear_queue()}.'
            if argument in ('pause', 'resume'):
                if argument == 'resume' and not any(self.archive.state(s)['auto_publish'] for s in self.routes):
                    return 'Публикация выключена. Сначала /post on.'
                for source in self.routes:
                    self.archive.set_delivery_enabled(source, argument == 'resume')
                return '⏳ Очередь запущена.' if argument == 'resume' else '⏸ Очередь остановлена.'
            if argument:
                return 'Используй /queue, /queue clear, /queue pause или /queue resume.'
            return '\n'.join(f'{self.archive.setting(f"route_name:{s}", f"-100{s}")}: '
                             f'{self.archive.queue_size(s)} ожидают' for s in self.routes) or 'Очередь пуста.'
        if command == '/an':
            if not argument:
                self.waiting.add(sender)
                return 'Пришли ссылку на пост, перешли пост или вставь текст. Отвечу здесь.'
            return await self.analyze(argument, sender=sender)
        async with self.command_lock:
            if command == '/channel':
                if len(argument.split()) != 2:
                    return 'Пришли /channel @источник @мой_канал'
                return await self.channel(*argument.split())
            if command in ('/post', '/index', '/llm'):
                if argument not in ('on', 'off'):
                    return f'Используй {command} on или {command} off.'
                enabled = argument == 'on'
                if command == '/llm':
                    await self.set_llm(enabled)
                    return f'🧠 MiniLM {"включена" if enabled else "выключена"}.'
                if not self.routes:
                    return 'Сначала настрой каналы: /channel @источник @мой_канал'
                for source, route in self.routes.items():
                    if command == '/post':
                        if enabled and not self.archive.state(source)['auto_publish']:
                            self.archive.set_posting_cursor(source, await self.telegram.head(route.source))
                        self.archive.set_auto_publish(source, int(enabled))
                        self.archive.set_delivery_enabled(source, int(enabled))
                    else:
                        self.archive.set_route_active(source, int(enabled))
                self.generation += 1
                label = 'Публикация новых постов' if command == '/post' else 'Индексация'
                return f'{label} {"включена" if enabled else "выключена"}.'
        return 'Неизвестная команда. Открой /help.' if command.startswith('/') else None

    async def handle(self, event):
        if not event.is_private or event.sender_id not in self.admins:
            return
        self.last_offer.pop(event.sender_id, None)
        raw = (event.raw_text or '').strip()
        try:
            if event.sender_id in self.waiting and getattr(event.message, 'fwd_from', None):
                self.waiting.discard(event.sender_id)
                answer = await self.analyze(raw, event.message.fwd_from, event.sender_id)
            elif raw.startswith('/'):
                answer = await self.dispatch(raw, event.sender_id)
            elif event.sender_id in self.waiting:
                self.waiting.discard(event.sender_id)
                answer = await self.analyze(raw, getattr(event.message, 'fwd_from', None), event.sender_id)
            else:
                return
        except Exception as exc:
            LOG.exception('Ошибка команды администратора')
            answer = f'⚠️ {type(exc).__name__}: {exc}'
        if answer:
            try:
                token = self.last_offer.pop(event.sender_id, None)
                buttons = [[Button.inline('➕ Индексировать', f'idx:{token}'.encode())]] if token else None
                await event.respond(answer[:4000], parse_mode=None, link_preview=False, buttons=buttons)
            except (OSError, errors.RPCError, RetryLater) as exc:
                LOG.warning('Ответ администратору не доставлен: %s', exc)

    async def handle_button(self, event):
        if not event.is_private or event.sender_id not in self.admins:
            return
        token = (event.data or b'').removeprefix(b'idx:').decode('ascii', errors='ignore')
        offer = self.index_offers.get(token) if (event.data or b'').startswith(b'idx:') else None
        if not offer or offer[0] != event.sender_id:
            await event.answer('Анализ устарел. Повтори /an.', alert=True)
            return
        try:
            added = self.archive.add_reference(*offer[1:])
            await event.answer('Добавлено в базу.' if added else 'Уже в базе.', alert=True)
        except Exception as exc:
            LOG.exception('Не удалось индексировать пост вручную')
            await event.answer(f'Ошибка: {type(exc).__name__}: {exc}'[:200], alert=True)


def admin_ids(raw):
    fields = [value for value in re.split(r'[,\s]+', raw.strip()) if value]
    if not fields or any(not re.fullmatch(r'[1-9][0-9]{0,18}', value) for value in fields):
        raise ValueError('Укажи числовые Telegram ID администраторов в TG_ADMIN_IDS')
    return frozenset(int(value) for value in fields)
