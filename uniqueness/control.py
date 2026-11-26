import asyncio
import logging
import re
from collections import defaultdict

from telethon import errors, types

from .analysis import segments_for
from .encoder import Encoder
from .telegram import RetryLater, channel_reference
from .worker import Route, deliver, index_batch

LOG = logging.getLogger(__name__)
HELP = """Управление анализом
/status — режим и маршруты
/route ИСТОЧНИК НАЗНАЧЕНИЕ — добавить, сменить или возобновить маршрут
/pause ИСТОЧНИК — остановить чтение и отправку
/drain ИСТОЧНИК — остановить чтение и отправить оставшуюся очередь
/mode minilm — включить MiniLM
/mode classic — выключить MiniLM

Канал можно указать как @username или https://t.me/username. Для приватного назначения нужен ID -100… ."""


class Control:
    def __init__(self, archive, telegram, model_dir, admins):
        self.archive = archive
        self.telegram = telegram
        self.model_dir = model_dir
        self.admins = admins
        self.mode = archive.setting('mode', 'minilm')
        self.encoder = None
        self.routes = {}
        self.locks = defaultdict(asyncio.Lock)
        self.commands = asyncio.Lock()
        self.generation = 0
        for source, values, active in archive.route_records():
            route = Route(*(types.InputPeerChannel(identifier, access_hash) for identifier, access_hash in values))
            self.routes[source] = (route, active)

    def status(self):
        mode = self.mode
        if mode == 'minilm' and self.encoder is None:
            mode += ' (ожидает загрузки модели)'
        yield f'Режим: {mode}'
        states = {0: 'пауза', 1: 'работает', 2: 'доставка очереди'}
        for row in self.archive.status_rows():
            source = row['source']
            pending = row['pending']
            if self.mode == 'minilm' and row['active'] == 1 and self.archive.missing_embedding(source):
                embedding = ', идёт заполнение эмбеддингов'
            else:
                embedding = ''
            history = 'готова' if row['history_done'] else 'индексируется'
            yield (f'-100{source} → -100{row["destination"]} · {states[row["active"]]} · '
                   f'история {history}\n'
                   f'В базе {row["indexed"]}, опубликовано {row["delivered"]}, '
                   f'в очереди {pending}, пропущено {row["skipped"]}{embedding}')

    async def set_mode(self, mode):
        if mode == 'minilm' and self.encoder is None:
            encoder = await asyncio.to_thread(Encoder, self.model_dir)
            self.archive.set_encoder(encoder)
            self.encoder = encoder
        if mode == 'classic':
            self.archive.set_encoder(None)
            self.encoder = None
        self.archive.set_setting('mode', mode)
        self.mode = mode
        self.generation += 1

    async def warm_model(self):
        async with self.commands:
            if self.mode == 'minilm' and self.encoder is None:
                await self.set_mode(self.mode)

    async def set_route(self, source_ref, destination_ref, *, skip_pending=False):
        route = await self.telegram.configure(source_ref, destination_ref)
        source = route.source.channel_id
        destination = route.destination.channel_id
        async with self.locks[source]:
            other = {identifier: old for identifier, (old, active) in self.routes.items()
                     if identifier != source and active}
            if destination in other or source in {item.destination.channel_id for item in other.values()}:
                raise ValueError('Активный канал не может быть одновременно источником и назначением')
            peers = [(peer.channel_id, peer.access_hash) for peer in
                     (route.source, route.destination, route.discussion, route.bot_discussion)]
            self.archive.save_route(source, destination, route.discussion.channel_id, peers,
                                    skip_pending=skip_pending)
            self.routes[source] = (route, 1)
            self.generation += 1
        return f'Маршрут работает: -100{source} → -100{destination}'

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
            route, _ = self.routes[source]
            self.routes[source] = (route, active)
            self.generation += 1
        return 'Чтение остановлено. Очередь будет отправлена.' if active else 'Маршрут на паузе.'

    async def index(self, source):
        route, active = self.routes[source]
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
        return await index_batch(self.archive, self.telegram, route.source, should_continue=
                                 lambda: generation == self.generation)

    async def deliver(self, source):
        async with self.locks[source]:
            route, active = self.routes[source]
            return await deliver(self.archive, self.telegram, route) if active else False

    async def dispatch(self, text):
        fields = text.split()
        if not fields:
            return []
        command, *args = fields
        command = command.lower()
        if command in ('/start', '/help') and not args:
            return [HELP]
        if command == '/status' and not args:
            return list(self.status())
        async with self.commands:
            if command == '/route' and len(args) in (2, 3):
                if len(args) == 2 or args[2] == 'skip_pending':
                    return [await self.set_route(*args[:2], skip_pending=len(args) == 3)]
            if command in ('/pause', '/drain') and len(args) == 1:
                return [await self.set_activity(args[0], 2 if command == '/drain' else 0)]
            if command == '/mode' and len(args) == 1 and args[0].lower() in ('classic', 'minilm'):
                mode = args[0].lower()
                await self.set_mode(mode)
                return [f'Режим: {mode}. Сохранённые оценки не изменены.']
        return ['Команда или параметры не распознаны. См. /help.'] if command.startswith('/') else []

    async def handle(self, event):
        if not event.is_private or event.sender_id not in self.admins:
            return
        try:
            lines = await self.dispatch(event.raw_text or '')
        except (ValueError, OSError, RuntimeError, RetryLater, errors.RPCError) as exc:
            lines = [f'Не удалось выполнить команду. {exc}']
        try:
            for offset in range(0, len(lines), 12):
                await event.respond('\n\n'.join(lines[offset:offset + 12]), parse_mode=None, link_preview=False)
        except (OSError, errors.RPCError) as exc:
            LOG.warning('Ответ администратору не доставлен: %s', exc)


def admin_ids(raw):
    fields = [value for value in re.split(r'[,\s]+', raw.strip()) if value]
    if not fields or any(not re.fullmatch(r'[1-9][0-9]{0,18}', value) for value in fields):
        raise ValueError('Укажи числовые Telegram ID администраторов в TG_ADMIN_IDS')
    return frozenset(int(value) for value in fields)
