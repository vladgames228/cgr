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

from telethon import errors, events

from .control import Control, admin_ids
from .storage import Archive, read_status
from .telegram import RetryLater, Telegram, client

LOG = logging.getLogger(__name__)


class Config:
    def __init__(self):
        self.data = Path(os.environ.get('DATA_DIR', './data'))
        self.data.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.model = os.environ.get('MODEL_DIR', '/model')

    def credentials(self):
        self.api_id = int(os.environ['TG_API_ID'])
        self.api_hash = os.environ['TG_API_HASH']


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


async def schedule(control, operation, idle, name):
    ready = {}
    failures = {}
    generation = -1
    while True:
        if generation != control.generation:
            generation = control.generation
            ready.clear()
            failures.clear()
        for source in tuple(control.routes):
            if ready.get(source, 0) > time.monotonic():
                continue
            try:
                work_remains = await operation(source)
            except (RetryLater, errors.RPCError, OSError, ValueError, asyncio.TimeoutError) as exc:
                failures[source] = failures.get(source, 0) + 1
                delay = min(120, 2 ** min(failures[source], 7))
                ready[source] = time.monotonic() + delay
                LOG.warning('%s, источник %s: %s. Повтор через %s с', name, source, exc, delay)
            else:
                failures.pop(source, None)
                ready[source] = time.monotonic() + (0 if work_remains else idle)
        now = time.monotonic()
        next_run = min(ready.values(), default=now + 1)
        await asyncio.sleep(max(0, min(1, next_run - now)))


async def serve(config, check=False):
    token = os.environ['TG_BOT_TOKEN']
    bot_id = int(token.split(':', 1)[0])
    poll = max(5, int(os.environ.get('POLL_SECONDS', '30')))
    admins = admin_ids(os.environ.get('TG_ADMIN_IDS', ''))
    archive = Archive(config.data / 'archive.sqlite3')
    user = client(config.data / 'reader', config.api_id, config.api_hash)
    bot = client(config.data / f'bot-{bot_id}', config.api_id, config.api_hash, updates=not check)
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
        archive.pin('reader_id', reader.id)
        archive.pin('bot_id', bot_id)
        control = Control(archive, telegram, config.model, admins)
        if check:
            await control.warm_model()
            return
        bot.add_event_handler(control.handle, events.NewMessage(incoming=True, func=lambda e: e.is_private))

        async def warm():
            try:
                await control.warm_model()
            except (ValueError, OSError, RuntimeError):
                LOG.exception('Не удалось загрузить MiniLM. Проверьте образ или переключитесь на /mode classic.')

        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(warm())
            tasks.create_task(schedule(control, control.index, poll, 'Чтение'))
            tasks.create_task(schedule(control, control.deliver, 2, 'Публикация'))
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
