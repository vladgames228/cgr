import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from telethon import errors, functions, types

from uniqueness.control import Control, admin_ids
from uniqueness.storage import Archive
from uniqueness.telegram import Telegram
from uniqueness.worker import ForwardRestricted, Route, deliver


def peer(identifier):
    return types.InputPeerChannel(identifier, 11)


def copy_message(identifier, text, media=None, grouped=None):
    return SimpleNamespace(id=identifier, message=text, media=media, grouped_id=grouped, invert_media=None,
                           entities=[types.MessageEntityBold(offset=0, length=len(text))] if text else None)


class UpdateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'archive.sqlite3'
        self.archive = Archive(self.path)
        self.addCleanup(lambda: self.archive.close())
        self.route = Route(peer(1), peer(2), peer(3), peer(3))

    def save_route(self):
        self.archive.save_route(1, 2, 3, [(i, 11) for i in (1, 2, 3, 3)])

    def reopen(self):
        self.archive.close()
        self.archive = Archive(self.path)

    async def test_protected_copy_restarts_without_another_forward(self):
        self.archive.register(1, 2, 3)
        self.archive.index(1, 7, '2025-01-01', 'Текст защищённого поста')
        api = SimpleNamespace(forward=AsyncMock(side_effect=ForwardRestricted),
                              copy=AsyncMock(side_effect=[ConnectionError(), 71]),
                              discussion_root=AsyncMock(return_value=81),
                              send_report=AsyncMock(return_value=91))
        with self.assertRaises(ConnectionError):
            await deliver(self.archive, api, self.route)
        self.reopen()
        self.assertTrue(await deliver(self.archive, api, self.route))
        self.assertEqual(api.copy.await_args_list[0], api.copy.await_args_list[1])
        api.forward.assert_awaited_once()
        api.discussion_root.assert_awaited_once_with(self.route, 71)
        self.assertEqual(api.send_report.await_args.args[1], 81)
        self.assertEqual(len(api.send_report.await_args.args[2].splitlines()), 4)

    async def test_text_is_copied_with_formatting(self):
        text = 'Текст защищённого поста'
        item = copy_message(7, text)
        user = AsyncMock(return_value=SimpleNamespace(updates=[types.UpdateMessageID(id=71, random_id=9)]))
        user.get_messages = AsyncMock(return_value=item)
        api = Telegram(user, None, self.archive)
        self.assertEqual(await api.copy(self.route, 7, 9), 71)
        request = user.await_args.args[0]
        self.assertIsInstance(request, functions.messages.SendMessageRequest)
        self.assertEqual(request.message, text)
        self.assertEqual(request.entities, item.entities)
        self.assertEqual(request.peer, self.route.destination)
        self.assertEqual(request.random_id, 9)

    async def test_media_copy_preserves_photo_caption(self):
        photo = types.MessageMediaPhoto(spoiler=True)
        item = copy_message(7, 'Подпись', photo)
        user = AsyncMock(return_value=SimpleNamespace(updates=[types.UpdateMessageID(id=71, random_id=9)]))
        user.get_messages = AsyncMock(return_value=item)
        user.download_media = AsyncMock(return_value='/tmp/test.jpg')
        user.upload_file = AsyncMock(return_value=types.InputFile(1, 1, 'test.jpg', ''))
        api = Telegram(user, None, self.archive)
        self.assertEqual(await api.copy(self.route, 7, 9), 71)
        request = user.await_args.args[0]
        self.assertIsInstance(request, functions.messages.SendMediaRequest)
        self.assertIsInstance(request.media, types.InputMediaUploadedPhoto)
        self.assertTrue(request.media.spoiler)
        self.assertEqual(request.message, item.message)
        self.assertEqual(request.entities, item.entities)

    async def test_album_preserves_uncaptioned_items_and_reuses_mapping(self):
        photo = types.MessageMediaPhoto(spoiler=True)
        first = copy_message(7, '', photo, 501)
        caption = copy_message(8, 'Подпись', photo, 501)

        async def requests(request):
            if isinstance(request, functions.messages.UploadMediaRequest):
                return types.MessageMediaPhoto(photo=types.Photo(
                    id=11, access_hash=12, file_reference=b'', date=None, sizes=[], dc_id=1))
            self.assertIsInstance(request, functions.messages.SendMultiMediaRequest)
            self.assertEqual([entry.message for entry in request.multi_media], ['', 'Подпись'])
            self.assertTrue(all(entry.media.spoiler for entry in request.multi_media))
            return SimpleNamespace(updates=[
                types.UpdateMessageID(80 + index, entry.random_id)
                for index, entry in enumerate(request.multi_media)
            ])

        user = AsyncMock(side_effect=requests)
        user.get_messages = AsyncMock(side_effect=[caption, [first, caption]])
        user.download_media = AsyncMock(return_value='/tmp/test.jpg')
        user.upload_file = AsyncMock(return_value=types.InputFile(1, 1, 'test.jpg', ''))
        api = Telegram(user, None, self.archive)
        self.assertEqual(await api.copy(self.route, 8, 19), 81)
        self.reopen()
        api = Telegram(user, None, self.archive)
        self.assertEqual(await api.forward(self.route, 7, 20), 80)
        self.assertEqual(user.await_count, 3)

    async def test_private_admin_only_and_settings_survive_restart(self):
        transport = SimpleNamespace(configure=AsyncMock(return_value=self.route),
                                    count=AsyncMock(return_value=12))
        control = Control(self.archive, transport, '/model', admin_ids('123, 456'))
        for identifier, private in ((77, True), (123, False)):
            event = SimpleNamespace(sender_id=identifier, is_private=private, raw_text='/mode classic',
                                    respond=AsyncMock())
            await control.handle(event)
            event.respond.assert_not_awaited()
        self.assertEqual(control.mode, 'classic')
        event = SimpleNamespace(sender_id=123, is_private=True, raw_text='/mode classic', respond=AsyncMock())
        await control.handle(event)
        event.respond.assert_awaited_once()
        await control.dispatch('/route @source_channel @destination_channel')
        await control.dispatch('/drain -1001')
        self.reopen()
        control = Control(self.archive, transport, '/model', {123})
        self.assertEqual(control.mode, 'classic')
        self.assertEqual(control.routes[1], (self.route, 2))
        await control.dispatch('/pause -1001')
        self.assertEqual(control.routes[1][1], 0)

    async def test_backfill_preserves_existing_scores_and_can_disable(self):
        self.save_route()
        self.archive.set_auto_publish(1, 1)
        self.archive.index(1, 7, '2025-01-01', 'Первое сообщение. И ещё один абзац.')
        previous = self.archive.next_job(1)
        self.archive.set_setting('mode', 'minilm')

        class Encoder:
            identity = 'mini-test'

            def encode(self, texts):
                return [[1, 2, 3] for _ in texts]

        control = Control(self.archive, None, '/model', {123})
        with patch('uniqueness.control.Encoder', return_value=Encoder()), \
             patch('uniqueness.control.ensure_model'):
            await control.warm_model()
        self.assertTrue(await control.index(1))
        self.assertIsNone(self.archive.missing_embedding(1))
        self.assertEqual(dict(previous), dict(self.archive.next_job(1)))
        await control.set_mode('classic')
        self.assertIsNone(self.archive.encoder)
        self.archive.index(1, 8, '2025-01-02', 'Новое сообщение без эмбеддинга.')
        self.assertEqual(self.archive.missing_embedding(1)['id'], 8)

    async def test_legacy_protected_posts_reenter_queue_only_once(self):
        self.archive.register(1, 2, 3)
        self.archive.index(1, 7, '2025-01-01', 'Текст защищённого поста.')
        self.archive.update_delivery(1, 7, skipped='Telegram запретил пересылку')
        self.save_route()
        self.assertTrue(self.archive.next_job(1)['copy_required'])
        with self.assertRaisesRegex(ValueError, 'очереди'):
            self.archive.save_route(1, 4, 5, [(i, 11) for i in (1, 4, 5, 5)])
        self.archive.update_delivery(1, 7, forward=100, result=200)
        self.archive.save_route(1, 4, 5, [(i, 11) for i in (1, 4, 5, 5)])
        self.assertIsNone(self.archive.next_job(1))
        self.assertEqual(self.archive.state(1)['cursor'], 7)

    async def test_explicit_route_switch_can_skip_undelivered_queue(self):
        self.save_route()
        self.archive.index(1, 7, '2025-01-01', 'Первое сообщение.')
        self.archive.update_delivery(1, 7, forward=88)
        self.archive.save_route(1, 4, 5, [(i, 11) for i in (1, 4, 5, 5)], skip_pending=True)
        self.assertIsNone(self.archive.next_job(1))
        self.assertEqual(self.archive.state(1)['cursor'], 7)
        self.archive.set_auto_publish(1, 1)
        self.archive.index(1, 8, '2025-01-02', 'Второе сообщение.')
        self.assertEqual(self.archive.next_job(1)['id'], 8)

    async def test_rpc_protection_is_recognized(self):
        user = AsyncMock(side_effect=errors.ChatForwardsRestrictedError(None))
        with self.assertRaises(ForwardRestricted):
            await Telegram(user, None, self.archive).forward(self.route, 7, 9)
