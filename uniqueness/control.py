import asyncio
import logging
import re
import time
from collections import defaultdict

from telethon import Button, errors, types

from .analysis import extract, index_groups, prepare_for_scoring, report, segments_for
from .encoder import Encoder, ensure_model
from .telegram import RetryLater, channel_reference
from .worker import Route, deliver, deliver_manual, index_batch

LOG = logging.getLogger(__name__)
HELP = """Панель управления
/route @источник @назначение — добавить маршрут и выбрать объём истории
/status — прогресс всех каналов и состояние модели
/queue — очередь и её остановка
/errors — последние ошибки
/llm on или /llm off — включить или выключить MiniLM
/pause @источник — остановить чтение и отправку
/drain @источник — закончить текущую очередь
/resume @источник — продолжить работу
/snooze @источник N — отложить комментарии на N минут (0 — отмена)
/analyze @источник ID — оценить конкретный пост без публикации
/summary @источник [N] — средние оценки уже обработанных постов
/autopost @источник on|off — автоматически публиковать новые проиндексированные посты
/cancel — отменить ввод числа или маршрута

Перешли боту пост из настроенного источника: он поставит его на публикацию и заново рассчитает оценку. Повторная пересылка создаст новый результат.

После /route выбери кнопкой: первые 100, первые 1000, всю историю или новые посты. Для иного числа используй /index @источник N. По умолчанию новые маршруты стоят на паузе."""


def response_chunks(lines):
    current = ''
    for line in lines:
        if current and len(current) + len(line) + 2 > 3500:
            yield current
            current = ''
        current = current + '\n\n' + line if current else line
    if current:
        yield current


class Control:
    def __init__(self, archive, telegram, model_dir, admins):
        self.archive = archive
        self.telegram = telegram
        self.model_dir = model_dir
        self.admins = admins
        self.mode = archive.setting('mode', 'classic')
        self.encoder = None
        self.model_state = 'проверка файлов'
        self.model_error = None
        self.last_alert = {}
        self.count_cache = {}
        self.manual_streak = defaultdict(int)
        self.pending_input = {}
        self.routes = {}
        self.locks = defaultdict(asyncio.Lock)
        self.commands = asyncio.Lock()
        self.model_lock = asyncio.Lock()
        self.generation = 0
        for source, values, active in archive.route_records():
            route = Route(*(types.InputPeerChannel(identifier, access_hash) for identifier, access_hash in values))
            self.routes[source] = (route, active)

    def status(self):
        mode = '🟢 MiniLM включена' if self.mode == 'minilm' and self.encoder else '⚪ MiniLM выключена'
        yield f'🧠 {mode}\n📦 Файлы модели: {self.model_state}'
        states = {0: 'пауза', 1: 'работает', 2: 'доставка очереди'}
        for row in self.archive.status_rows():
            source = row['source']
            pending = self.archive.queue_size(source)
            if self.mode == 'minilm' and row['active'] == 1 and self.archive.missing_embedding(source):
                embedding = ', идёт заполнение эмбеддингов'
            else:
                embedding = ''
            history = 'готова' if row['history_done'] else 'индексируется'
            cap = f' · лимит {row["index_limit"]}' if row['index_limit'] is not None else ''
            snooze = f' · ответы отложены до {time.strftime("%H:%M", time.localtime(row["reply_until"]))}' if row['reply_until'] > time.time() else ''
            problem = f'\n⚠️ {self.archive.error(source)}' if self.archive.error(source) else ''
            total = self.count_cache.get(source, (None, 0))[0]
            total_line = f'📨 в канале около {total} сообщений · ' if total is not None else ''
            yield (f'📣 -100{source} → -100{row["destination"]}\n'
                   f'🔹 {states[row["active"]]} · отправка {"вкл" if row["delivery_enabled"] else "выкл"}'
                   f' · автопубликация {"вкл" if row["auto_publish"] else "выкл"} · история {history}{cap}{snooze}\n'
                   f'{total_line}📚 просмотрено {row["scan_count"]}, с текстом {row["indexed"]}\n'
                   f'✅ опубликовано {row["delivered"]} · ⏳ очередь {pending} · '
                   f'⚠️ пропущено {row["skipped"]}{embedding}{problem}')

    async def set_mode(self, mode):
        if mode == 'minilm' and self.encoder is None:
            try:
                await self.ensure_ready()
                encoder = await asyncio.to_thread(Encoder, self.model_dir)
                self.archive.set_encoder(encoder)
                self.encoder = encoder
                self.model_state = 'готовы'
                self.model_error = None
            except Exception as exc:
                self.model_state = 'ошибка'
                self.model_error = str(exc)
                raise
        if mode == 'classic':
            self.archive.set_encoder(None)
            self.encoder = None
        self.archive.set_setting('mode', mode)
        self.mode = mode
        self.generation += 1

    async def ensure_ready(self):
        async with self.model_lock:
            self.model_state = 'загрузка'
            try:
                await asyncio.to_thread(ensure_model, self.model_dir)
            except Exception as exc:
                self.model_state = 'ошибка'
                self.model_error = str(exc)
                raise
            self.model_state = 'готовы'
            self.model_error = None

    async def warm_model(self):
        try:
            await self.ensure_ready()
            async with self.commands:
                if self.mode == 'minilm' and self.encoder is None:
                    await self.set_mode(self.mode)
        except Exception as exc:
            self.model_state = 'ошибка'
            self.model_error = str(exc)
            raise

    async def set_route(self, source_ref, destination_ref, *, skip_pending=False):
        route = await self.telegram.configure(source_ref, destination_ref)
        source = route.source.channel_id
        destination = route.destination.channel_id
        count = await self.telegram.count(route.source)
        self.count_cache[source] = (count, time.monotonic())
        async with self.locks[source]:
            other = {identifier: old for identifier, (old, active) in self.routes.items()
                     if identifier != source and active}
            if destination in other or source in {item.destination.channel_id for item in other.values()}:
                raise ValueError('Активный канал не может быть одновременно источником и назначением')
            peers = [(peer.channel_id, peer.access_hash) for peer in
                     (route.source, route.destination, route.discussion, route.bot_discussion)]
            active = 0
            self.archive.save_route(source, destination, route.discussion.channel_id, peers,
                                    skip_pending=skip_pending, active=active)
            self.archive.set_delivery_enabled(source, 1)
            self.routes[source] = (route, active)
            self.generation += 1
        indexed = self.archive.indexed_count(source)
        return (f'📣 -100{source} → -100{destination}\n'
                f'📨 В канале около {count} сообщений; в базе {indexed} постов с текстом; '
                f'в очереди {self.archive.queue_size(source)}.\n'
                f'🧠 Режим: {"MiniLM" if self.mode == "minilm" else "классический"}. '
                'Индексация на паузе; ручные публикации принимаются. Выбери объём истории.')

    async def source_id(self, reference):
        ref = channel_reference(reference)
        if isinstance(ref, int):
            return int(str(ref)[4:])
        entity = await self.telegram.call('user', self.telegram.user.get_input_entity, ref)
        if not isinstance(entity, types.InputPeerChannel):
            raise ValueError('Укажи источник по ссылке, username или ID')
        return entity.channel_id

    async def set_activity(self, reference, active):
        source = await self.source_id(reference)
        async with self.locks[source]:
            self.archive.set_route_active(source, active)
            self.archive.set_delivery_enabled(source, active != 0)
            route, _ = self.routes[source]
            self.routes[source] = (route, active)
            self.generation += 1
        return {0: '⏸ Чтение и очередь остановлены.',
                1: '▶️ Чтение и отправка возобновлены.',
                2: '📤 Чтение остановлено; очередь доставляется.'}[active]

    async def start_index(self, reference, amount):
        source = await self.source_id(reference)
        if source not in self.routes:
            raise ValueError('Сначала добавь маршрут через /route')
        async with self.locks[source]:
            route, _ = self.routes[source]
            if amount == 'new':
                head = await self.telegram.head(route.source)
                self.archive.start_from_now(source, head)
                label = f'Новые публикации после {head}'
            else:
                limit = None if amount == 'all' else int(amount)
                if limit is not None and not 1 <= limit <= 1000000:
                    raise ValueError('Укажи число от 1 до 1000000')
                state = self.archive.state(source)
                self.archive.set_index_limit(source, None if limit is None else state['scan_count'] + limit)
                label = 'Вся история' if limit is None else f'Следующие {limit} сообщений'
            self.archive.set_route_active(source, 1)
            self.archive.set_delivery_enabled(source, 1)
            self.routes[source] = (route, 1)
            self.generation += 1
        return f'▶️ {label}. Прогресс смотри в /status.'

    async def analyze_post(self, reference, post):
        source = await self.source_id(reference)
        if source not in self.routes:
            raise ValueError('Источник не настроен. Сначала используй /route')
        route = self.routes[source][0]
        item = await self.telegram.call('user', self.telegram.user.get_messages, route.source, ids=post)
        if item is None or not item.message:
            raise ValueError('Пост не найден или в нём нет текста')
        features = extract(item.message, self.encoder)
        probes = list(index_groups(features, query=True))
        scores = self.archive.score(source, post, prepare_for_scoring(features), probes)
        return f'Пост {post}, сравнение с ранее проиндексированными:\n' + report(scores)

    async def alert(self, source, operation, exc):
        message = f'{operation}: {type(exc).__name__}: {exc}'
        self.archive.remember_error(source, message)
        key = (source, operation, type(exc).__name__, str(exc))
        now = time.monotonic()
        if now - self.last_alert.get(key, 0) < 600:
            return
        self.last_alert[key] = now
        for admin in self.admins:
            try:
                await self.telegram.call('bot', self.telegram.bot.send_message, admin,
                                         f'⚠️ Канал -100{source}\n{message[:700]}\nОткрой /status или /errors.')
            except (OSError, errors.RPCError, RetryLater):
                LOG.warning('Не удалось отправить уведомление администратору %s', admin)

    async def refresh_counts(self):
        for source, (route, _) in self.routes.items():
            if time.monotonic() - self.count_cache.get(source, (None, 0))[1] < 120:
                continue
            try:
                self.count_cache[source] = (await self.telegram.count(route.source), time.monotonic())
            except (OSError, errors.RPCError, RetryLater) as exc:
                LOG.warning('Не удалось получить число сообщений в %s: %s', source, exc)

    async def notify_index_done(self, source):
        count, scores = self.archive.summary(source)
        state = self.archive.state(source)
        text = (f'✅ Индексация -100{source} завершила выбранный объём. '
                f'Просмотрено {state["scan_count"]} сообщений, в базе {count} постов с текстом.\n'
                f'Средние сохранённые оценки:\n{report(scores)}\n/status · /summary -100{source}')
        for admin in self.admins:
            try:
                await self.telegram.call('bot', self.telegram.bot.send_message, admin, text)
            except (OSError, errors.RPCError, RetryLater) as exc:
                LOG.warning('Уведомление об индексации не доставлено: %s', exc)

    async def index(self, source):
        route, active = self.routes[source]
        stored = self.archive.db.execute('SELECT active FROM route_settings WHERE source = ?',
                                         (source,)).fetchone()[0]
        if stored != active:
            self.routes[source] = (route, stored)
            self.generation += 1
            active = stored
        if active != 1:
            return False
        generation = self.generation
        if self.mode == 'minilm':
            if self.encoder is None:
                return False
            if row := self.archive.missing_embedding(source):
                features = self.archive.unpack_features(row['features'])
                if 'vector' not in features['units'][0]:
                    segments = segments_for(row['original'])[3]
                    encoder = self.encoder
                    vectors = await asyncio.to_thread(lambda: list(encoder.encode(segments)))
                    for item, vector in zip(features['units'], vectors, strict=True):
                        item['vector'] = vector
                self.archive.store_embedding(source, row['id'], features)
                return True
        history_before = self.archive.state(source)['history_done']
        remains = await index_batch(self.archive, self.telegram, route.source, should_continue=
                                    lambda: generation == self.generation)
        stored = self.archive.db.execute('SELECT active FROM route_settings WHERE source = ?',
                                         (source,)).fetchone()[0]
        if stored != active:
            self.routes[source] = (route, stored)
            self.generation += 1
        state_after = self.archive.state(source)
        limit_done = (state_after['index_limit'] is not None and
                      state_after['scan_count'] >= state_after['index_limit'])
        if limit_done or (not history_before and state_after['history_done']):
            await self.notify_index_done(source)
        return remains

    async def deliver(self, source):
        async with self.locks[source]:
            route, active = self.routes[source]
            if not self.archive.delivery_enabled(source):
                return False
            if self.manual_streak[source] >= 3 and await deliver(self.archive, self.telegram, route):
                self.manual_streak[source] = 0
                return True
            if await deliver_manual(self.archive, self.telegram, route):
                self.manual_streak[source] += 1
                return True
            self.manual_streak[source] = 0
            return await deliver(self.archive, self.telegram, route)

    async def dispatch(self, text):
        fields = text.split()
        if not fields:
            return []
        command, *args = fields
        command = command.lower()
        if command in ('/start', '/help') and not args:
            return [HELP]
        if command == '/cancel' and not args:
            return ['Ввод отменён. Открой /start для панели.']
        if command == '/status' and not args:
            await self.refresh_counts()
            return list(self.status())
        if command == '/errors' and not args:
            problems = [f'-100{source}: {self.archive.error(source)}' for source in self.routes
                        if self.archive.error(source)]
            if self.model_error:
                problems.insert(0, f'Модель: {self.model_error}')
            return problems or ['✅ Последних ошибок нет.']
        if command == '/queue' and not args:
            return [f'📣 -100{source}: ⏳ {self.archive.queue_size(source)} · '
                    f'{"отправка включена" if self.archive.delivery_enabled(source) else "остановлена"}'
                    for source in self.routes] or ['Очередь пуста: маршрутов пока нет.']
        if command == '/analyze-help' and len(args) == 1:
            return [f'🔎 Отправь /analyze {args[0]} ID_ПОСТА. Например, '
                    f'/analyze {args[0]} 123. Для публикации просто перешли этот пост боту.']
        async with self.commands:
            if command == '/route' and len(args) in (2, 3):
                if len(args) == 2 or args[2] == 'skip_pending':
                    return [await self.set_route(*args[:2], skip_pending=len(args) == 3)]
            if command in ('/pause', '/drain', '/resume') and len(args) == 1:
                return [await self.set_activity(args[0], {'/pause': 0, '/drain': 2, '/resume': 1}[command])]
            if command in ('/stopqueue', '/startqueue') and len(args) == 1:
                source = await self.source_id(args[0])
                self.archive.set_delivery_enabled(source, command == '/startqueue')
                return [f'📤 Очередь -100{source}: {"запущена" if command == "/startqueue" else "остановлена"}.']
            if command == '/autopost' and len(args) == 2 and args[1] in ('on', 'off'):
                source = await self.source_id(args[0])
                self.archive.set_auto_publish(source, args[1] == 'on')
                return [f'📣 Автопубликация новых постов -100{source}: {"включена" if args[1] == "on" else "выключена"}.']
            if command == '/index' and len(args) == 2:
                return [await self.start_index(args[0], args[1].lower())]
            if command == '/snooze' and len(args) == 2:
                minutes = int(args[1])
                if not 0 <= minutes <= 10080:
                    raise ValueError('Число минут должно быть от 0 до 10080')
                source = await self.source_id(args[0])
                self.archive.set_reply_until(source, time.time() + minutes * 60 if minutes else 0)
                return [f'⏰ Комментарии для -100{source} отложены на {minutes} мин.' if minutes else
                        f'▶️ Комментарии для -100{source} снова разрешены.']
            if command in ('/llm', '/mode') and len(args) == 1 and args[0].lower() in ('on', 'off', 'classic', 'minilm'):
                mode = 'minilm' if args[0].lower() in ('on', 'minilm') else 'classic'
                await self.set_mode(mode)
                return [f'🧠 MiniLM {"включена" if mode == "minilm" else "выключена"}. Сохранённые оценки не изменены.']
            if command == '/summary' and len(args) in (1, 2):
                source = await self.source_id(args[0])
                limit = int(args[1]) if len(args) == 2 else None
                if limit is not None and not 1 <= limit <= 1000000:
                    raise ValueError('Укажи число от 1 до 1000000')
                count, scores = self.archive.summary(source, limit)
                return [f'📊 -100{source}: {count} постов с текстом. Средние оценки на момент индексации:\n{report(scores)}']
            if command == '/analyze' and len(args) == 2:
                return [await self.analyze_post(args[0], int(args[1]))]
        return ['Команда или параметры не распознаны. См. /help.'] if command.startswith('/') else []

    def buttons(self, text, lines=None):
        command = text.split()[0].lower() if text.split() else ''
        if command in ('/start', '/help', '/status', '/queue'):
            rows = [[Button.inline('➕ Добавить маршрут', b'ask:route')],
                    [Button.inline('📊 Статус', b'cmd:/status'),
                     Button.inline('⏳ Очередь', b'cmd:/queue')],
                    [Button.inline('🧠 MiniLM вкл.', b'cmd:/llm on'),
                     Button.inline('⚪ MiniLM выкл.', b'cmd:/llm off')],
                    [Button.inline('⚠️ Ошибки', b'cmd:/errors'),
                     Button.inline('❔ Помощь', b'cmd:/help')]]
            for source in list(self.routes)[:8]:
                rows.append([Button.inline(f'⏸ -100{source}', f'cmd:/pause -100{source}'.encode()),
                             Button.inline('▶️', f'cmd:/resume -100{source}'.encode()),
                             Button.inline('📤', f'cmd:/drain -100{source}'.encode())])
                rows.append([Button.inline('⏰ 15 мин', f'cmd:/snooze -100{source} 15'.encode()),
                             Button.inline('⏰ 60 мин', f'cmd:/snooze -100{source} 60'.encode()),
                             Button.inline('📊 Аналитика', f'cmd:/summary -100{source}'.encode())])
                rows.append([Button.inline('🚀 Авто вкл.', f'cmd:/autopost -100{source} on'.encode()),
                             Button.inline('Авто выкл.', f'cmd:/autopost -100{source} off'.encode()),
                             Button.inline('🛑 Очередь', f'cmd:/stopqueue -100{source}'.encode())])
                rows.append([Button.inline('▶️ Очередь', f'cmd:/startqueue -100{source}'.encode()),
                             Button.inline('🔎 Оценить пост', f'ask:analyze:{source}'.encode())])
            return rows
        if command == '/index':
            fields = text.split()
            if len(fields) == 3:
                rows = [[Button.inline('📊 Статус', b'cmd:/status'),
                         Button.inline('📈 Аналитика', f'cmd:/summary {fields[1]}'.encode())]]
                if re.fullmatch(r'-100[0-9]+', fields[1]):
                    rows.append([Button.inline('📈 Первые N', f'ask:summary:{fields[1][4:]}'.encode()),
                                 Button.inline('🔎 Оценить пост', f'ask:analyze:{fields[1][4:]}'.encode())])
                return rows
        if command == '/route':
            match = re.match(r'📣 -100(\d+) →', lines[0]) if lines else None
            if match:
                source = int(match[1])
                return [[Button.inline('Первые 100', f'cmd:/index -100{source} 100'.encode()),
                         Button.inline('Первые 1000', f'cmd:/index -100{source} 1000'.encode())],
                        [Button.inline('Вся история', f'cmd:/index -100{source} all'.encode()),
                         Button.inline('Только новые', f'cmd:/index -100{source} new'.encode())],
                        [Button.inline('📈 Аналитика базы', f'cmd:/summary -100{source}'.encode()),
                         Button.inline('🔎 Оценить пост', f'ask:analyze:{source}'.encode())],
                        [Button.inline('Своё число N', f'ask:index:{source}'.encode()),
                         Button.inline('📈 Первые N', f'ask:summary:{source}'.encode())],
                        [Button.inline('📊 Статус', b'cmd:/status')]]
        return None

    async def handle(self, event):
        if not event.is_private or event.sender_id not in self.admins:
            return
        forward = getattr(getattr(event, 'message', None), 'fwd_from', None)
        if forward and (not isinstance(forward.from_id, types.PeerChannel) or not forward.channel_post):
            await event.respond('⚠️ Telegram не указал исходный канал или ID поста. '
                                'Перешли обычный пост из настроенного канала либо используй /analyze @канал ID.',
                                parse_mode=None)
            return
        if forward and isinstance(forward.from_id, types.PeerChannel) and forward.channel_post:
            source = forward.from_id.channel_id
            if source not in self.routes:
                await event.respond(f'Источник -100{source} не настроен. Сначала используй /route.',
                                    parse_mode=None)
                return
            try:
                result = await self.analyze_post(f'-100{source}', forward.channel_post)
                self.archive.add_manual(event.sender_id, event.id, source, forward.channel_post,
                                        result.split('\n', 1)[1])
                empty = self.archive.indexed_count(source) == 0
                warning = '⚠️ База источника пока пуста: сначала проиндексируй историю для сравнения.\n' if empty else ''
                await event.respond(f'✅ Пост {forward.channel_post} проанализирован и поставлен в очередь.\n'
                                    f'{warning}'
                                    f'{result}\n\nОчередь: /queue', parse_mode=None)
            except Exception as exc:
                LOG.exception('Ошибка ручной публикации')
                await event.respond(f'⚠️ Не удалось обработать пересланный пост: {exc}', parse_mode=None)
            return
        text = event.raw_text or ''
        if text.strip().lower() == '/cancel':
            self.pending_input.pop(event.sender_id, None)
        pending = self.pending_input.get(event.sender_id)
        if pending and not text.startswith('/'):
            action, source = pending
            if action == 'route':
                if len(text.split()) != 2:
                    await event.respond('Пришли два адреса через пробел: @источник @назначение.', parse_mode=None)
                    return
                text = '/route ' + text
            else:
                if not re.fullmatch(r'[1-9][0-9]{0,6}', text.strip()):
                    await event.respond('Пришли положительное число до 1000000.', parse_mode=None)
                    return
                text = f'/{action} -100{source} {text.strip()}'
            self.pending_input.pop(event.sender_id, None)
        try:
            lines = await self.dispatch(text)
        except Exception as exc:
            LOG.exception('Ошибка команды администратора')
            lines = [f'Не удалось выполнить команду. {exc}']
        try:
            for offset, chunk in enumerate(response_chunks(lines)):
                await event.respond(chunk, parse_mode=None,
                                    link_preview=False,
                                    buttons=self.buttons(text, lines) if offset == 0 else None)
        except (OSError, errors.RPCError) as exc:
            LOG.warning('Ответ администратору не доставлен: %s', exc)

    async def handle_button(self, event):
        if event.sender_id not in self.admins or not event.is_private:
            return
        data = event.data.decode('utf-8', errors='replace')
        if data.startswith('ask:'):
            _, action, *rest = data.split(':')
            if action not in ('route', 'index', 'summary', 'analyze'):
                return
            source = int(rest[0]) if rest else None
            self.pending_input[event.sender_id] = (action, source)
            await event.answer()
            prompt = {'route': 'Пришли @источник @назначение одним сообщением.',
                      'index': f'Сколько следующих сообщений канала -100{source} проиндексировать?',
                      'summary': f'Для скольких первых постов канала -100{source} показать аналитику?',
                      'analyze': f'Пришли ID поста канала -100{source} для новой оценки.'}[action]
            await event.respond(prompt, parse_mode=None)
            return
        if not data.startswith('cmd:'):
            return
        command = data[4:]
        try:
            await event.answer('Выполняю…')
        except errors.RPCError:
            pass
        try:
            lines = await self.dispatch(command)
        except Exception as exc:
            LOG.exception('Ошибка кнопки панели')
            lines = [f'⚠️ {exc}']
        for offset, chunk in enumerate(response_chunks(lines)):
            await event.respond(chunk, parse_mode=None, link_preview=False,
                                buttons=self.buttons(command, lines) if offset == 0 else None)


def admin_ids(raw):
    fields = [value for value in re.split(r'[,\s]+', raw.strip()) if value]
    if not fields or any(not re.fullmatch(r'[1-9][0-9]{0,18}', value) for value in fields):
        raise ValueError('Укажи числовые Telegram ID администраторов в TG_ADMIN_IDS')
    return frozenset(int(value) for value in fields)
