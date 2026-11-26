from pathlib import Path
from tempfile import TemporaryDirectory

from telethon import functions, types, utils

from .analysis import stable_hash
from .worker import Undeliverable

FILE_MEDIA = (types.MessageMediaPhoto, types.MessageMediaDocument)


async def upload_payload(telegram, item):
    media = item.media
    if not isinstance(media, FILE_MEDIA) or media.ttl_seconds:
        raise Undeliverable(f'тип вложения нельзя воспроизвести: {type(media).__name__}')
    user = telegram.user
    # Stream one file at a time without retaining media in RAM or the database.
    with TemporaryDirectory(prefix='telegram-post-') as temporary:
        downloaded = await telegram.call('user', user.download_media, item, file=temporary + '/')
        if not downloaded:
            raise Undeliverable('Telegram не предоставил файл вложения')
        file = await telegram.call('user', user.upload_file, Path(downloaded))
    if isinstance(media, types.MessageMediaPhoto):
        return types.InputMediaUploadedPhoto(file=file, spoiler=media.spoiler)
    document = media.document
    streamed = (types.DocumentAttributeVideo, types.DocumentAttributeAudio, types.DocumentAttributeAnimated)
    return types.InputMediaUploadedDocument(
        file=file, mime_type=document.mime_type, attributes=document.attributes,
        spoiler=media.spoiler, force_file=not any(isinstance(attr, streamed) for attr in document.attributes),
    )


async def send_album(telegram, route, items, post, random_id):
    album = []
    ids = {}
    for item in items:
        payload = await upload_payload(telegram, item)
        uploaded = await telegram.call('user', telegram.user, functions.messages.UploadMediaRequest(
            peer=route.destination, media=payload))
        media = utils.get_input_media(uploaded)
        media.spoiler = item.media.spoiler
        identifier = random_id if item.id == post else stable_hash(f'{random_id}:album:{item.id}') or 1
        ids[item.id] = identifier
        album.append(types.InputSingleMedia(
            media=media, message=item.message or '', entities=item.entities, random_id=identifier))
    updates = await telegram.call('user', telegram.user, functions.messages.SendMultiMediaRequest(
        peer=route.destination, multi_media=album, silent=True))
    return updates, ids


async def send_copy(telegram, route, post, random_id):
    user = telegram.user
    item = await telegram.call('user', user.get_messages, route.source, ids=post)
    if not item:
        raise Undeliverable('публикация удалена или недоступна')
    if item.grouped_id:
        # Album items are adjacent channel messages, with up to ten items per album.
        neighbors = await telegram.call('user', user.get_messages, route.source,
                                        ids=list(range(max(1, post - 9), post + 10)))
        items = sorted((entry for entry in neighbors if entry and getattr(entry, 'grouped_id', None) == item.grouped_id),
                       key=lambda entry: entry.id)
        if len(items) > 1 and any(entry.id == post for entry in items):
            return await send_album(telegram, route, items, post, random_id)
    arguments = {
        'peer': route.destination,
        'message': item.message or '',
        'entities': item.entities,
        'random_id': random_id,
        'silent': True,
        'invert_media': item.invert_media,
    }
    media = item.media
    if media is None or isinstance(media, (types.MessageMediaEmpty, types.MessageMediaWebPage)):
        request = functions.messages.SendMessageRequest(
            no_webpage=not isinstance(media, types.MessageMediaWebPage), **arguments)
    else:
        payload = await upload_payload(telegram, item)
        request = functions.messages.SendMediaRequest(media=payload, **arguments)
    return await telegram.call('user', user, request), {post: random_id}
