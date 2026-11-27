import asyncio
import re
import time

from telethon import TelegramClient, errors, functions, types

from .copy_post import send_copy
from .worker import ForwardRestricted, Incoming, Route, Undeliverable

LINK = re.compile(r'(?:https?://(?:t\.me|telegram\.me)/(?:s/)?)?@?([A-Za-z][A-Za-z0-9_]{3,31})/?')
NUMERIC = re.compile(r'-100[0-9]+')
FORWARD_GAP = 1.25
REPORT_GAP = 3.1


class RetryLater(Exception):
    pass


def channel_reference(value, public=False):
    if isinstance(value, str) and (match := LINK.fullmatch(value.strip())):
        return '@' + match[1]
    if not public and isinstance(value, (str, int)) and NUMERIC.fullmatch(str(value)):
        return int(value)
    raise ValueError(f'Некорректная ссылка или идентификатор канала: {value!r}')


def client(session, api_id, api_hash, *, updates=False):
    return TelegramClient(str(session), api_id, api_hash, flood_sleep_threshold=0,
                          request_retries=1, raise_last_call_error=True,
                          entity_cache_limit=500, receive_updates=updates)


def sent_id(updates, random_id, target, *, single=True):
    if single and isinstance(updates, types.UpdateShortSentMessage):
        return updates.id
    items = (updates.update,) if isinstance(updates, types.UpdateShort) else getattr(updates, 'updates', ())
    for update in items:
        if isinstance(update, types.UpdateMessageID) and update.random_id == random_id:
            return update.id
    messages = [
        update.message for update in items
        if isinstance(update, (types.UpdateNewMessage, types.UpdateNewChannelMessage))
        and update.message.peer_id == types.PeerChannel(target.channel_id)
    ]
    if single and len(messages) == 1:
        return messages[0].id
    raise RetryLater('Telegram пока не вернул идентификатор отправленного сообщения')


class Telegram:
    def __init__(self, user, archive):
        self.user = user
        self.archive = archive
        self.last_forwards = 0.0
        self.last_reports = {}

    def pause_remaining(self, identity):
        return max(0, self.archive.flood_until(identity) - time.time())

    async def call(self, identity, operation, *args, **kwargs):
        if wait := self.pause_remaining(identity):
            raise RetryLater(f'Telegram просит подождать ещё {int(wait) + 1} с')
        try:
            return await operation(*args, **kwargs)
        except (errors.FloodWaitError, errors.FloodPremiumWaitError, errors.SlowModeWaitError) as exc:
            self.archive.defer_flood(identity, time.time() + exc.seconds + 1)
            raise RetryLater(f'Ограничение Telegram на {exc.seconds} с') from exc

    async def configure(self, source_ref, destination_ref):
        source_name = channel_reference(source_ref, public=True)
        destination_name = channel_reference(destination_ref)
        source = await self.call('user', self.user.get_entity, source_name)
        if not isinstance(source, types.Channel) or not source.broadcast:
            raise ValueError(f'Источник {source_name} должен быть публичным каналом')
        try:
            destination = await self.call('user', self.user.get_entity, destination_name)
        except ValueError:
            if not isinstance(destination_name, int):
                raise
            dialogs = self.user.iter_dialogs()
            destination = None
            while destination is None:
                try:
                    dialog = await self.call('user', anext, dialogs)
                except StopAsyncIteration:
                    break
                if dialog.id == destination_name:
                    destination = dialog.entity
        if not isinstance(destination, types.Channel) or not destination.broadcast:
            raise ValueError(f'Назначение {destination_name} должно быть каналом')
        if source.id == destination.id:
            raise ValueError('Канал назначения совпадает с источником')
        if not (destination.creator or getattr(destination.admin_rights, 'post_messages', False)):
            raise ValueError(f'Пользовательской сессии нужно право публикации в {destination_name}')
        return Route(await self.user.get_input_entity(source),
                     await self.user.get_input_entity(destination))

    async def configure_group(self, reference):
        if isinstance(reference, str) and reference.startswith('-100') and reference[4:].isdigit():
            group_id = int(reference[4:])
            entity = None
            async for dialog in self.user.iter_dialogs():
                if isinstance(dialog.entity, types.Channel) and dialog.entity.id == group_id:
                    entity = dialog.entity
                    break
        else:
            entity = await self.call('user', self.user.get_entity, reference)
        if not entity or not isinstance(entity, (types.Chat, types.Channel)) or (
                isinstance(entity, types.Channel) and entity.broadcast):
            raise ValueError('Группа недоступна твоему аккаунту или это канал')
        return entity, await self.user.get_input_entity(entity)

    async def head(self, source):
        messages = await self.call('user', self.user.get_messages, source, limit=1)
        return messages[0].id if messages else 0

    async def count(self, source):
        messages = await self.call('user', self.user.get_messages, source, limit=0)
        return messages.total

    async def history(self, source, after, head, limit):
        iterator = self.user.iter_messages(source, min_id=after, max_id=head + 1,
                                           reverse=True, limit=limit)
        while True:
            try:
                item = await self.call('user', anext, iterator)
            except StopAsyncIteration:
                return
            if not after < item.id <= head:
                raise RuntimeError(f'Telegram вернул сообщение вне диапазона истории: {item.id}')
            after = item.id
            yield Incoming(item.id, item.date, getattr(item, 'message', None))

    async def pace_posts(self):
        await asyncio.sleep(max(0, self.last_forwards + FORWARD_GAP - time.monotonic()))
        self.last_forwards = time.monotonic()

    async def copy(self, route, post, random_id, *, reuse_copy=True):
        source, destination = route.source.channel_id, route.destination.channel_id
        existing = self.archive.copied_id(source, destination, post) if reuse_copy else None
        if existing is not None:
            return existing
        await self.pace_posts()
        updates, ids = await send_copy(self, route, post, random_id)
        copies = {original: sent_id(updates, identifier, route.destination, single=len(ids) == 1)
                  for original, identifier in ids.items()}
        self.archive.store_copies(source, destination, copies)
        return copies[post]

    async def forward(self, route, post, random_id, *, reuse_copy=True):
        existing = (self.archive.copied_id(route.source.channel_id, route.destination.channel_id, post)
                    if reuse_copy else None)
        if existing is not None:
            return existing
        await self.pace_posts()
        try:
            updates = await self.call('user', self.user, functions.messages.ForwardMessagesRequest(
                from_peer=route.source, id=[post], random_id=[random_id], to_peer=route.destination,
                silent=True,
            ))
        except errors.ChatForwardsRestrictedError as exc:
            raise ForwardRestricted from exc
        except (errors.MessageIdInvalidError, errors.MsgIdInvalidError) as exc:
            messages = await self.call('user', self.user.get_messages, route.source, ids=post)
            if messages is None:
                raise Undeliverable('публикация удалена или недоступна') from exc
            raise
        return sent_id(updates, random_id, route.destination)

    async def send_report(self, route, forwarded_post, report, random_id, original=None):
        channel_id = route.destination.channel_id
        deadline = self.last_reports.get(channel_id, 0) + REPORT_GAP
        await asyncio.sleep(max(0, deadline - time.monotonic()))
        self.last_reports[channel_id] = time.monotonic()
        quote = ''
        for character in original or '':
            if len((quote + character).encode('utf-8')) > 900:
                break
            quote += character
        reply = types.InputReplyToMessage(reply_to_msg_id=forwarded_post,
                                          quote_text=quote or None, quote_offset=0 if quote else None)
        try:
            updates = await self.call('user', self.user, functions.messages.SendMessageRequest(
                peer=route.destination, message=report, random_id=random_id, silent=True,
                no_webpage=True, reply_to=reply,
            ))
        except (errors.MsgIdInvalidError, errors.MessageIdInvalidError) as exc:
            raise Undeliverable('пересланный пост недоступен для цитирования') from exc
        return sent_id(updates, random_id, route.destination)

    async def send_text(self, route, text, random_id):
        await self.pace_posts()
        updates = await self.call('user', self.user, functions.messages.SendMessageRequest(
            peer=route.destination, message=text, random_id=random_id, silent=True, no_webpage=True))
        return sent_id(updates, random_id, route.destination)
