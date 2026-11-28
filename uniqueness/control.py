import asyncio
import json
import logging
import os
import re
import secrets
import time

from telethon import Button, errors, types

from .analysis import words
from .encoder import Encoder, ensure_model
from .telegram import RetryLater, channel_reference
from .worker import ForwardRestricted, Route, Undeliverable, deliver, index_batch

LOG = logging.getLogger(__name__)
POST_LINK = re.compile(r'(?:https?://)?t\.me/(?:s/)?([A-Za-z][A-Za-z0-9_]{3,31})/(\d+)(?:\?.*)?')
PRIVATE_LINK = re.compile(r'(?:https?://)?t\.me/c/(\d+)/(\d+)(?:\?.*)?')
HELP = (
    'Бот собирает историю канала в базу. Публикация новых постов включается отдельно.\n\n'
    '/channel @источник @мой_канал — выбрать каналы\n'
    '/channel remove @источник — убрать канал из наблюдения, сохранив индекс\n'
    '/group add @группа ID — следить за сообщениями автора\n'
    '/group — список; /group recent ID — последние оценки; /group remove ID — убрать\n'
    '/minwords N — минимум слов для индексации (по умолчанию 3)\n'
    '/post on|off @источник — публикация новых постов\n'
    '/index on|off @источник — индексация; /index rebuild @источник — пересборка\n'
    '/llm on|off — включить или выключить MiniLM\n'
    'Для проверки пришли ссылку, перешли пост или отправь текст\n'
    '/reverse on|off — поменять направление процентов\n'
    '/label текст — изменить подпись к проценту\n'
    '/stat — частые слова и выражения в базе\n'
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
        self.groups = {}
        self.count_cache = {}
        self.index_offers = {}
        self.last_offer = {}
        self.posting_offers = set()
        self.posted_offers = set()
        self.post_progress = {}
        self.model_ram_mb = None
        self.command_lock = asyncio.Lock()
        self.delivery_locks = {}
        self.generation = 0
        for source, peers, _ in archive.route_records():
            self.routes[source] = Route(*(types.InputPeerChannel(*peer) for peer in peers[:2]))
        for row in archive.groups():
            kind, peer_id, access_hash = json.loads(row['peer'])
            self.groups[row['id']] = (types.InputPeerChannel(peer_id, access_hash)
                                      if kind == 'channel' else types.InputPeerChat(peer_id))

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
            before_ram = self.ram_mb()
            try:
                await asyncio.to_thread(ensure_model, self.model_dir)
                encoder = await asyncio.to_thread(Encoder, self.model_dir)
                self.archive.set_encoder(encoder)
                self.encoder = encoder
                self.model_state = 'готова'
                after_ram = self.ram_mb()
                self.model_ram_mb = max(0, after_ram - before_ram) if before_ram is not None and after_ram is not None else None
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
            self.model_ram_mb = None
        self.mode = 'minilm' if enabled else 'classic'
        self.archive.set_setting('mode', self.mode)
        self.generation += 1

    async def channel(self, source_ref, destination_ref):
        route = await self.telegram.configure(source_ref, destination_ref)
        source = route.source.channel_id
        destination = route.destination.channel_id
        head = await self.telegram.head(route.source)
        count = await self.telegram.count(route.source)
        old = self.archive.state(source)
        publish_new = any(self.archive.state(item)['auto_publish'] for item in self.routes)
        peers = [(peer.channel_id, peer.access_hash) for peer in (route.source, route.destination)]
        self.archive.save_route(source, destination, peers, active=old['active'] if old else 1)
        if old is None:
            self.archive.set_route_active(source, 1)
            self.archive.set_auto_publish(source, publish_new)
            self.archive.set_delivery_enabled(source, publish_new)
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

    async def selected_sources(self, reference):
        if not reference:
            return list(self.routes)
        source = await self.source_id(reference)
        if source not in self.routes:
            raise ValueError('Источник не настроен. Список: /channel')
        return [source]

    async def group_command(self, argument):
        parts = argument.split()
        if not parts:
            rows = self.archive.groups()
            return ('\n'.join(f'{row["name"]} · ID {row["id"]} · автор {row["user_id"]} · '
                              f'{"вкл" if row["active"] else "выкл"} · в базе '
                              f'{self.archive.indexed_count(row["id"])}'
                              for row in rows) or 'Групп пока нет. /group add @группа ID_автора')
        action = parts[0].lower()
        if action == 'add' and len(parts) == 3 and parts[2].isdigit():
            entity, peer = await self.telegram.configure_group(parts[1])
            if entity.id in self.routes:
                raise ValueError('Эта группа уже используется как канал')
            kind = 'channel' if isinstance(peer, types.InputPeerChannel) else 'chat'
            descriptor = [kind, entity.id, getattr(peer, 'access_hash', 0)]
            head = await self.telegram.head(peer)
            self.archive.save_group(entity.id, descriptor, entity.title, int(parts[2]), head)
            self.groups[entity.id] = peer
            self.generation += 1
            return f'👥 {entity.title}: слежу за автором {parts[2]}. Начинаю с новых сообщений.'
        if action in ('remove', 'on', 'off', 'recent') and len(parts) == 2 and parts[1].isdigit():
            group_id = int(parts[1])
            row = self.archive.group(group_id)
            if row is None or row['active'] < 0:
                return 'Группа не найдена. Список: /group'
            if action == 'recent':
                items = self.archive.recent_group_results(group_id)
                linkable = json.loads(row['peer'])[0] == 'channel'
                return ('\n'.join(f'{result} · '
                                  f'{f"https://t.me/c/{group_id}/{post}" if linkable else f"сообщение {post}"}'
                                  for post, result in items) or 'Оценок пока нет.')
            if action == 'remove':
                self.archive.remove_group(group_id)
                self.groups.pop(group_id, None)
            else:
                with self.archive.db as db:
                    db.execute('UPDATE groups SET active = ? WHERE id = ?', (int(action == 'on'), group_id))
            self.generation += 1
            return f'👥 {row["name"]}: {"убрана" if action == "remove" else action}.'
        return 'Используй /group add @группа ID_автора; /group recent ID; /group on|off|remove ID.'

    async def watch_group(self, group_id):
        row = self.archive.group(group_id)
        if not row or not row['active']:
            return False
        peer = self.groups[group_id]
        head = await self.telegram.head(peer)
        if head <= row['cursor']:
            if self.encoder:
                return self.archive.fill_vectors(group_id)
            return False
        iterator = self.telegram.user.iter_messages(peer, min_id=row['cursor'], max_id=head + 1,
                                                    reverse=True, limit=40)
        count = 0
        while True:
            try:
                item = await self.telegram.call('user', anext, iterator)
            except StopAsyncIteration:
                break
            text = getattr(item, 'message', '') or ''
            if item.sender_id == row['user_id'] and len(words(text)) >= int(self.archive.setting('min_words', 3)):
                result = self.score_text(group_id, text)
                self.archive.add_reference(group_id, item.id, text, encode=False)
                self.archive.group_result(group_id, item.id, result)
            self.archive.group_cursor(group_id, item.id)
            count += 1
        return count == 40

    def score_text(self, source, text, exclude=None, *, detailed=False):
        if not text or not words(text):
            return 'У поста нет слов для оценки.'
        scores = self.archive.analyze(source, text, exclude)
        reverse = self.archive.setting('reverse', 'off') == 'on'
        place = 'группе' if self.archive.group(source) else 'канале'
        label = self.archive.setting('label', '') or (f'Повторяемость в {place}' if reverse else f'Уникальность в {place}')
        value = 100 - scores['overall'] if reverse else scores['overall']
        result = f'{label}: {value}%'
        if detailed:
            direction = 'Повтор' if reverse else 'Новизна'
            content = 100 - scores['content'] if reverse else scores['content']
            result += f'\n{direction} содержания: {content}%'
            if scores['style'] is not None:
                style = 100 - scores['style'] if reverse else scores['style']
                result += f'\n{direction} стиля: {style}%'
            else:
                result += '\nСтиль: нужно 8 слов в тексте и 20 образцов в архиве'
            if scores['model_pending']:
                result += '\nСмысловая модель ещё обрабатывает архив'
            elif self.archive.encoder is None:
                result += '\nСмысл: включи /llm on для сравнения перефразировок'
        if detailed and scores['similarity'] > 60:
            similar_source, post = scores['similar_post']
            name = self.archive.setting(f'route_name:{similar_source}', '').split(' → ', 1)[0]
            link = ('ручной образец' if post < 0 else
                    f'https://t.me/{name[1:]}/{post}' if name.startswith('@') else
                    f'https://t.me/c/{similar_source}/{post}')
            result += f'\nПохожий пост ({scores["similarity"]}%): {link}'
        return result

    @staticmethod
    def ram_mb():
        try:
            with open('/proc/self/status', encoding='ascii') as status:
                for line in status:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) / 1024
        except OSError:
            pass
        return None

    def offer_index(self, sender, source, post, text, result):
        if sender is None or not text or not words(text):
            return
        token = secrets.token_urlsafe(12)
        self.index_offers[token] = (sender, source, post, text, result)
        self.last_offer[sender] = token
        if len(self.index_offers) > 100:
            expired = next(iter(self.index_offers))
            self.index_offers.pop(expired)
            self.post_progress.pop(expired, None)
            self.posted_offers.discard(expired)

    async def analyze(self, value=None, forward=None, sender=None):
        if not self.routes and not self.archive.statistics(0)[0]:
            raise ValueError('Сначала выбери каналы: /channel @источник @мой_канал')
        if (forward and isinstance(forward.from_id, types.PeerChannel)
                and forward.channel_post):
            source = forward.from_id.channel_id
            post = forward.channel_post
            text = (await self.telegram.call('user', self.telegram.user.get_messages,
                                             self.routes[source].source, ids=post)
                    if source in self.routes else None)
            original = text.message if text else self.archive.post_text(source, post)
            if original is not None:
                result = self.score_text(source, original, post, detailed=True)
                if source in self.routes:
                    self.offer_index(sender, source, post, original, result.split('\n', 1)[0])
                return result
        value = (value or '').strip()
        public = POST_LINK.fullmatch(value)
        private = PRIVATE_LINK.fullmatch(value)
        if public or private:
            match = public or private
            reference = '@' + public[1] if public else '-100' + private[1]
            source = await self.source_id(reference)
            post = int(match[2])
            item = (await self.telegram.call('user', self.telegram.user.get_messages,
                                             self.routes[source].source, ids=post)
                    if source in self.routes else None)
            original = item.message if item else self.archive.post_text(source, post)
            if original is None:
                raise ValueError('Пост не найден в сохранённой базе')
            result = self.score_text(source, original, post, detailed=True)
            if source in self.routes:
                self.offer_index(sender, source, post, original, result.split('\n', 1)[0])
            return result
        if value.startswith(('t.me/', 'https://t.me/', 'http://t.me/')):
            raise ValueError('Пришли ссылку на отдельный пост, например https://t.me/channel/123')
        if not value:
            return None
        source = None
        if value.startswith('@') and ' ' in value:
            reference, value = value.split(' ', 1)
            source = await self.source_id(reference)
        elif self.routes:
            source = next(iter(self.routes))
        if source is None and self.archive.statistics(0)[0]:
            return self.score_text(0, value, detailed=True)
        if source not in self.routes:
            raise ValueError('Для текста укажи канал: @канал текст')
        result = self.score_text(source, value, detailed=True)
        self.offer_index(sender, source, None, value, result.split('\n', 1)[0])
        return result

    async def index(self, source):
        if self.mode == 'minilm' and self.encoder is None:
            return False
        if source not in self.routes or not self.archive.route_active(source):
            return False
        generation = self.generation
        if self.encoder and self.archive.fill_vectors(source):
            return True
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

    async def capture(self, event):
        peer = getattr(event.message, 'peer_id', None)
        if not isinstance(peer, types.PeerChannel):
            return
        source = peer.channel_id
        text = getattr(event.message, 'message', None)
        if (source not in self.routes or not self.archive.route_active(source) or not text
                or len(words(text)) < int(self.archive.setting('min_words', 3))):
            return
        try:
            self.archive.add_reference(source, event.message.id, text, encode=False)
        except Exception as exc:
            LOG.exception('Не удалось сохранить новый пост %s/%s', source, event.message.id)
            self.archive.remember_error(source, f'Новый пост: {type(exc).__name__}: {exc}')

    async def deliver(self, source):
        if self.mode == 'minilm' and self.encoder is None:
            return False
        state = self.archive.state(source)
        if not state or not state['auto_publish'] or not self.archive.delivery_enabled(source):
            return False
        lock = self.delivery_locks.setdefault(source, asyncio.Lock())
        async with lock:
            return await deliver(self.archive, self.telegram, self.routes[source],
                                 should_continue=lambda: bool(self.archive.state(source)
                                                             and self.archive.state(source)['auto_publish']
                                                             and self.archive.delivery_enabled(source)))

    async def status(self):
        lines = [f'📤 Публикация: {"вкл" if any(self.archive.state(s)["auto_publish"] for s in self.routes) else "выкл"}',
                 f'📚 Индексация: {"вкл" if any(self.archive.route_active(s) for s in self.routes) else "выкл"}',
                 f'🧠 MiniLM: {"вкл" if self.mode == "minilm" and self.encoder else "выкл"}']
        if self.model_ram_mb is not None and self.encoder:
            lines.append(f'🧠 Память MiniLM: ~{self.model_ram_mb:.0f} МБ')
        if ram := self.ram_mb():
            lines.append(f'💾 Память процесса: {ram:.0f} МБ')
        lines.append(f'🗃 База: {self.archive.size_mb():.1f} МБ; общий индекс каналов: '
                     f'{self.archive.statistics(0)[0]} разных текстов')
        if self.model_error:
            lines.append(f'⚠️ Модель: {self.model_error[:250]}')
        reverse = self.archive.setting('reverse', 'off') == 'on'
        label = self.archive.setting('label', '') or ('Повторяемость в канале' if reverse else 'Уникальность в канале')
        lines.append(f'📊 {label}: 100% = {"повтор" if reverse else "новый текст"}')
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
            ready = self.archive.vector_count(source) if self.encoder else None
            without_text = max(0, row['scan_count'] - indexed)
            new = max(0, cached - row['scan_count']) if cached is not None else '?'
            name = self.archive.setting(f'route_name:{source}', f'-100{source}')
            lines.append(f'\n{name}\n📨 в канале: {cached if cached is not None else "?"}; '
                         f'просмотрено: {row["scan_count"]}; в базе: {indexed}; без текста/служебных: ~{without_text}; осталось просмотреть: ~{new}\n'
                         f'⏳ очередь: {self.archive.queue_size(source)} '
                         f'({"идёт" if self.archive.delivery_enabled(source) and row["auto_publish"] else "пауза"})')
            if ready is not None:
                lines.append(f'🧠 Векторы MiniLM: {ready}/{indexed}')
            if error := self.archive.error(source):
                lines.append(f'⚠️ {error[:250]}')
        if not self.routes:
            lines.append('Каналы ещё не настроены. Начни с /channel @источник @мой_канал')
        if self.groups:
            lines.append(f'👥 Группы: {len(self.groups)}; минимум слов: {self.archive.setting("min_words", 3)}')
        return '\n'.join(lines)

    async def dispatch(self, text, sender=None):
        command, _, argument = text.strip().partition(' ')
        command = command.lower().split('@', 1)[0]
        argument = argument.strip()
        if command == '/stop':
            self.generation += 1
            self.archive.stop_all()
            return '🛑 Публикация, индексация и очередь остановлены.'
        if command in ('/start', '/help'):
            return HELP
        if command == '/status':
            return await self.status()
        if command == '/stat':
            lines = []
            total, terms, expressions = self.archive.statistics(0)
            if total:
                lines.append(f'Все каналы вместе: {total} разных текстов в базе')
                lines.append('Частые основы слов (число постов): ' +
                             (', '.join(f'{word} — {count}' for word, count in terms) or 'пока нет'))
                lines.append('Повторяющиеся выражения: ' +
                             (', '.join(f'{" ".join(phrase)} — {count}'
                                        for phrase, count in expressions) or 'пока нет'))
            for source in self.groups:
                total, terms, expressions = self.archive.statistics(source)
                group = self.archive.group(source)
                name = group['name'] if group else self.archive.setting(f'route_name:{source}', f'-100{source}').split(' → ', 1)[0]
                lines.append(f'{name}: {total} постов в базе')
                lines.append('Частые основы слов (число постов): ' +
                             (', '.join(f'{word} — {count}' for word, count in terms) or 'пока нет'))
                lines.append('Повторяющиеся выражения: ' +
                             (', '.join(f'{" ".join(phrase)} — {count}'
                                        for phrase, count in expressions if count > 1) or 'пока нет'))
                if self.encoder:
                    lines.append(f'MiniLM: {self.archive.vector_count(source)}/{total} векторов')
            return '\n'.join(lines) or 'База пока пуста.'
        if command == '/reverse':
            if argument not in ('on', 'off'):
                return 'Используй /reverse on или /reverse off.'
            self.archive.set_setting('reverse', argument)
            return 'Теперь 100% означает повтор.' if argument == 'on' else 'Теперь 100% означает новый текст.'
        if command == '/label':
            if not argument:
                return 'Используй /label Новый текст или /label reset.'
            if len(argument) > 80 or '\n' in argument or '\r' in argument:
                return 'Подпись должна быть одной строкой до 80 символов.'
            self.archive.set_setting('label', '' if argument == 'reset' else argument)
            return 'Подпись сброшена.' if argument == 'reset' else f'Подпись: {argument}'
        if command == '/minwords':
            if not argument:
                return f'Минимум слов для новых сообщений: {self.archive.setting("min_words", 3)}. /minwords N'
            if not argument.isdigit() or not 1 <= int(argument) <= 100:
                return 'Укажи число от 1 до 100: /minwords 5'
            self.archive.apply_min_words(int(argument))
            return f'Теперь индексируются сообщения от {argument} слов. Старые записи сохранены.'
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
        async with self.command_lock:
            if command == '/group':
                return await self.group_command(argument)
            if command == '/channel':
                parts = argument.split()
                if not parts:
                    return ('\n'.join(f'{self.archive.setting(f"route_name:{source}", f"-100{source}")}: '
                                      f'индекс {"вкл" if self.archive.route_active(source) else "выкл"}, '
                                      f'постинг {"вкл" if self.archive.state(source)["auto_publish"] else "выкл"}'
                                      for source in self.routes) or 'Каналов пока нет.')
                if len(parts) == 2 and parts[0].lower() == 'remove':
                    source = (await self.selected_sources(parts[1]))[0]
                    self.archive.remove_route(source)
                    self.routes.pop(source)
                    self.count_cache.pop(source, None)
                    self.generation += 1
                    return 'Источник убран из наблюдения. Его проиндексированные тексты остались в общей базе.'
                if len(parts) != 2:
                    return 'Пришли /channel @источник @мой_канал или /channel remove @источник'
                return await self.channel(*argument.split())
            if command == '/index' and argument.split()[:1] == ['rebuild']:
                if not self.routes:
                    return 'Сначала настрой каналы: /channel @источник @мой_канал'
                parts = argument.split()
                if len(parts) > 2 or (len(parts) == 1 and len(self.routes) > 1):
                    return 'Укажи источник: /index rebuild @канал'
                sources = await self.selected_sources(parts[1] if len(parts) == 2 else '')
                self.generation += 1
                for source in sources:
                    self.archive.rebuild_index(source)
                return '📚 Выбранный источник пересобирается. Остальные тексты общей базы сохранены.'
            if command in ('/post', '/index', '/llm'):
                parts = argument.split()
                if not parts or parts[0] not in ('on', 'off') or len(parts) > 2 or (command == '/llm' and len(parts) > 1):
                    return ('Используй /index on, /index off или /index rebuild.' if command == '/index'
                            else f'Используй {command} on или {command} off.')
                enabled = parts[0] == 'on'
                if command == '/llm':
                    await self.set_llm(enabled)
                    return f'🧠 MiniLM {"включена" if enabled else "выключена"}.'
                if not self.routes:
                    return 'Сначала настрой каналы: /channel @источник @мой_канал'
                sources = await self.selected_sources(parts[1] if len(parts) == 2 else '')
                for source in sources:
                    route = self.routes[source]
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
            if getattr(event.message, 'fwd_from', None):
                answer = await self.analyze(raw, event.message.fwd_from, event.sender_id)
            elif raw.startswith('/'):
                answer = await self.dispatch(raw, event.sender_id)
            else:
                answer = await self.analyze(raw, sender=event.sender_id) if raw else None
        except Exception as exc:
            LOG.exception('Ошибка команды администратора')
            answer = f'⚠️ {type(exc).__name__}: {exc}'
        if answer:
            try:
                token = self.last_offer.pop(event.sender_id, None)
                buttons = [[Button.inline('➕ Индексировать', f'idx:{token}'.encode()),
                            Button.inline('📤 Запостить', f'post:{token}'.encode())]] if token else None
                await event.respond(answer[:4000], parse_mode=None, link_preview=False, buttons=buttons)
            except (OSError, errors.RPCError, RetryLater) as exc:
                LOG.warning('Ответ администратору не доставлен: %s', exc)

    async def handle_button(self, event):
        if not event.is_private or event.sender_id not in self.admins:
            return
        data = event.data or b''
        action, _, raw_token = data.partition(b':')
        token = raw_token.decode('ascii', errors='ignore')
        offer = self.index_offers.get(token) if action in (b'idx', b'post') else None
        if not offer or offer[0] != event.sender_id:
            await event.answer('Анализ устарел. Отправь текст или ссылку ещё раз.', alert=True)
            return
        try:
            if action == b'idx':
                added = self.archive.add_reference(*offer[1:4])
                await event.answer('Добавлено в базу.' if added else 'Уже в базе.', alert=True)
            else:
                if token in self.posted_offers:
                    await event.answer('Уже опубликовано.', alert=True)
                    return
                if token in self.posting_offers:
                    await event.answer('Публикация уже идёт.', alert=True)
                    return
                self.posting_offers.add(token)
                try:
                    _, source, post, text, result = offer
                    route = self.routes[source]
                    random_post, random_report, published, draft = self.post_progress.setdefault(
                        token, [secrets.randbits(63) or 1, secrets.randbits(63) or 1, None, None])
                    if published is None:
                        if post is None:
                            published = await self.telegram.send_text(route, text, random_post)
                        else:
                            try:
                                try:
                                    published = await self.telegram.forward(route, post, random_post)
                                except ForwardRestricted:
                                    published = await self.telegram.copy(route, post, random_post)
                            except Undeliverable:
                                published = await self.telegram.send_text(route, text, random_post)
                        self.post_progress[token][2] = published
                    if draft is None:
                        draft = await self.telegram.prepare_report(route, published, result, random_report, text)
                        self.post_progress[token][3] = draft
                    await self.telegram.publish_report(route, draft, random_report)
                    self.posted_offers.add(token)
                    self.post_progress.pop(token, None)
                finally:
                    self.posting_offers.discard(token)
                await event.answer('Опубликовано в канале.', alert=True)
        except Exception as exc:
            LOG.exception('Не удалось индексировать пост вручную')
            await event.answer(f'Ошибка: {type(exc).__name__}: {exc}'[:200], alert=True)


def admin_ids(raw):
    fields = [value for value in re.split(r'[,\s]+', raw.strip()) if value]
    if not fields or any(not re.fullmatch(r'[1-9][0-9]{0,18}', value) for value in fields):
        raise ValueError('Укажи числовые Telegram ID администраторов в TG_ADMIN_IDS')
    return frozenset(int(value) for value in fields)
