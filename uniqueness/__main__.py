import argparse
import asyncio
import fcntl
import json
import logging
import os
import signal
import sqlite3
import time
from pathlib import Path

from telethon import errors

from .storage import Archive, read_status
from .telegram import RetryLater, Telegram, client
from .worker import deliver, index_batch

LOG = logging.getLogger(__name__)


class Config:
    def __init__(self):
        self.data = Path(os.environ.get('DATA_DIR', './data'))
        self.data.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.model = os.environ.get('MODEL_DIR')

    def credentials(self):
        self.api_id = int(os.environ['TG_API_ID'])
        self.api_hash = os.environ['TG_API_HASH']

    def routes(self):
        raw = Path(os.environ.get('CHANNELS_FILE', './channels.json')).read_text(encoding='utf-8')
        routes = json.loads(raw)
        if not isinstance(routes, list) or not routes:
            raise ValueError('channels.json должен содержать непустой массив маршрутов')
        for route in routes:
            if not isinstance(route, dict) or set(route) != {'source', 'destination'}:
                raise ValueError('Каждый маршрут должен содержать поля source и destination')
        return routes

    def archive(self):
        encoder = None
        if self.model:
            from .encoder import Encoder
            encoder = Encoder(self.model)
        return Archive(self.data / 'archive.sqlite3', encoder.identity if encoder else 'classic-v1', encoder)


async def login(config):
    user = client(config.data / 'reader', config.api_id, config.api_hash)
    try:
        await user.start()
        me = await user.get_me()
        if me.bot:
            raise ValueError('Нужна пользовательская сессия Telegram')
        LOG.info('Пользовательская сессия сохранена. ID аккаунта %s', me.id)
    finally:
        await user.disconnect()


async def schedule(routes, operation, idle, name):
    ready = [0.0] * len(routes)
    failures = [0] * len(routes)
    while True:
        for index, route in enumerate(routes):
            if ready[index] > time.monotonic():
                continue
            try:
                work_remains = await operation(route)
            except (RetryLater, errors.RPCError, OSError, asyncio.TimeoutError) as exc:
                failures[index] += 1
                delay = min(120, 2 ** min(failures[index], 7))
                ready[index] = time.monotonic() + delay
                LOG.warning('%s, источник %s: %s. Повтор через %s с',
                            name, route.source.channel_id, exc, delay)
            else:
                failures[index] = 0
                ready[index] = time.monotonic() + (0 if work_remains else idle)
        await asyncio.sleep(max(0, min(ready) - time.monotonic()))


async def serve(config, check=False):
    token = os.environ['TG_BOT_TOKEN']
    bot_id = int(token.split(':', 1)[0])
    poll = max(5, int(os.environ.get('POLL_SECONDS', '30')))
    settings = config.routes()
    archive = config.archive()
    user = client(config.data / 'reader', config.api_id, config.api_hash)
    bot = client(config.data / f'bot-{bot_id}', config.api_id, config.api_hash)
    try:
        telegram = Telegram(user, bot, archive)
        while wait := max(telegram.pause_remaining('user'), telegram.pause_remaining('bot')):
            LOG.info('Пауза Telegram до повторного подключения, осталось %d с', wait + 1)
            await asyncio.sleep(wait)
        await telegram.call('user', user.connect)
        reader = await telegram.call('user', user.get_me)
        if not reader or reader.bot:
            raise ValueError('Сначала авторизуйте пользовательскую сессию командой login')
        await telegram.call('bot', bot.start, bot_token=token)
        if (await telegram.call('bot', bot.get_me)).id != bot_id:
            raise ValueError('ID сессии бота не совпадает с токеном')
        routes = []
        for setting in settings:
            routes.append(await telegram.configure(setting['source'], setting['destination']))
        sources = {route.source.channel_id for route in routes}
        if len(sources) != len(routes) or sources & {route.destination.channel_id for route in routes}:
            raise ValueError('Источник можно указать один раз. Назначение должно быть отдельным каналом')
        archive.pin('reader_id', reader.id)
        archive.pin('bot_id', bot_id)
        for route in routes:
            archive.register(route.source.channel_id, route.destination.channel_id, route.discussion.channel_id)
            LOG.info('Маршрут %s → %s, обсуждения %s', route.source.channel_id,
                     route.destination.channel_id, route.discussion.channel_id)
        if check:
            return
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(schedule(routes, lambda route: index_batch(archive, telegram, route.source),
                                       poll, 'Чтение'))
            tasks.create_task(schedule(routes, lambda route: deliver(archive, telegram, route),
                                       2, 'Публикация'))
    finally:
        await asyncio.gather(user.disconnect(), bot.disconnect())
        archive.close()


async def run_with_signals(operation):
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(operation)
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        await task
    except asyncio.CancelledError:
        pass
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description='Накопительный анализ Telegram-публикаций')
    parser.add_argument('command', nargs='?', default='run', choices=('login', 'check', 'run', 'status'))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logging.getLogger('telethon').setLevel(logging.WARNING)
    try:
        config = Config()
        if args.command == 'status':
            for row in read_status(config.data / 'archive.sqlite3'):
                print(json.dumps(row, ensure_ascii=False))
            return 0
        with (config.data / 'worker.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError('В этом каталоге данных уже работает экземпляр приложения') from exc
            config.credentials()
            job = login(config) if args.command == 'login' else serve(config, check=args.command == 'check')
            asyncio.run(run_with_signals(job))
    except (KeyError, ValueError, OSError, RuntimeError, sqlite3.Error, RetryLater, errors.RPCError) as exc:
        LOG.error('%s', exc)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
