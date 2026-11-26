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
    def __init__(self, user, bot, archive):
        self.user = user
        self.bot = bot
        self.archive = archive
        self.last_forwards = 0.0
        self.last_reports = {}
        self.last_bot_send = 0.0

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
        full = await self.call('user', self.user, functions.channels.GetFullChannelRequest(destination))
        group_id = full.full_chat.linked_chat_id
        if not group_id:
            raise ValueError(f'К каналу {destination_name} нужно привязать группу обсуждений')
        discussion = next((chat for chat in full.chats if chat.id == group_id and
                           isinstance(chat, types.Channel) and chat.megagroup), None)
        if not discussion or discussion.left:
            raise ValueError(f'Пользовательской сессии нужно вступить в группу обсуждений {group_id}')
        try:
            bot_group = await self.call('bot', self.bot.get_input_entity, types.PeerChannel(group_id))
            membership = await self.call('bot', self.bot, functions.channels.GetParticipantRequest(
                bot_group, types.InputPeerSelf()))
        except (ValueError, errors.ChannelPrivateError, errors.UserNotParticipantError) as exc:
            raise ValueError(f'Добавьте бота в группу обсуждений {group_id}') from exc
        if isinstance(membership.participant, (types.ChannelParticipantLeft, types.ChannelParticipantBanned)):
            raise ValueError(f'Боту нужен доступ к отправке сообщений в группе {group_id}')
        return Route(await self.user.get_input_entity(source),
                     await self.user.get_input_entity(destination),
                     await self.user.get_input_entity(discussion), bot_group)

    async def head(self, source):
        messages = await self.call('user', self.user.get_messages, source, limit=1)
        return messages[0].id if messages else 0

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

    async def copy(self, route, post, random_id):
        source, destination = route.source.channel_id, route.destination.channel_id
        existing = self.archive.copied_id(source, destination, post)
        if existing is not None:
            return existing
        await self.pace_posts()
        updates, ids = await send_copy(self, route, post, random_id)
        copies = {original: sent_id(updates, identifier, route.destination, single=len(ids) == 1)
                  for original, identifier in ids.items()}
        self.archive.store_copies(source, destination, copies)
        return copies[post]

    async def forward(self, route, post, random_id):
        existing = self.archive.copied_id(route.source.channel_id, route.destination.channel_id, post)
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

    async def discussion_root(self, route, forward):
        try:
            result = await self.call('user', self.user, functions.messages.GetDiscussionMessageRequest(
                peer=route.destination, msg_id=forward))
        except (errors.MsgIdInvalidError, errors.MessageIdInvalidError) as exc:
            post = await self.call('user', self.user.get_messages, route.destination, ids=forward)
            if post is None:
                raise Undeliverable('пересланный пост удалён') from exc
            raise RetryLater(f'Нет ветки обсуждения для пересланного поста {forward}') from exc
        if not result.messages:
            raise RetryLater(f'Telegram ещё не создал ветку для поста {forward}')
        root = result.messages[-1]
        if root.peer_id != types.PeerChannel(route.discussion.channel_id):
            raise ValueError('Группа обсуждений назначения изменилась')
        return root.id

    async def send_report(self, route, root, report, random_id):
        group_id = route.bot_discussion.channel_id
        deadline = max(self.last_reports.get(group_id, 0) + REPORT_GAP, self.last_bot_send + 0.06)
        await asyncio.sleep(max(0, deadline - time.monotonic()))
        self.last_reports[group_id] = self.last_bot_send = time.monotonic()
        try:
            updates = await self.call('bot', self.bot, functions.messages.SendMessageRequest(
                peer=route.bot_discussion, message=report, random_id=random_id, silent=True,
                no_webpage=True, reply_to=types.InputReplyToMessage(reply_to_msg_id=root),
            ))
        except errors.BadRequestError as exc:
            missing_parent = isinstance(exc, (errors.MsgIdInvalidError, errors.MessageIdInvalidError))
            if not missing_parent and exc.message != 'TOPIC_ID_INVALID':
                raise
            parent = await self.call('user', self.user.get_messages, route.discussion, ids=root)
            if parent is None:
                raise Undeliverable('ветка обсуждения удалена') from exc
            raise
        return sent_id(updates, random_id, route.bot_discussion)
