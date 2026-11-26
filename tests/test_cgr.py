import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from telethon import functions, types

from uniqueness.control import Control
from uniqueness.storage import Archive
from uniqueness.telegram import Telegram
from uniqueness.worker import Incoming, Route, deliver, index_batch


def peer(identifier):
    return types.InputPeerChannel(identifier, 11)


def item(identifier, text):
    return Incoming(identifier, datetime(2025, 1, 1, tzinfo=timezone.utc), text)


class CgrTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.archive = Archive(Path(self.temp.name) / 'db.sqlite3')
        self.addCleanup(self.archive.close)
        self.archive.save_route(1, 2, [(1, 11), (2, 11)])
        self.route = Route(peer(1), peer(2))

    def control(self, telegram):
        return Control(self.archive, telegram, self.temp.name, {100})

    async def test_new_channel_starts_passive_indexing(self):
        fresh = Archive(Path(self.temp.name) / 'fresh.sqlite3')
        self.addCleanup(fresh.close)
        telegram = SimpleNamespace(configure=AsyncMock(return_value=self.route),
                                   head=AsyncMock(return_value=50), count=AsyncMock(return_value=45))
        control = Control(fresh, telegram, self.temp.name, {100})
        answer = await control.dispatch('/channel @source_channel @my_channel')
        self.assertIn('Индексация включена, публикация выключена', answer)
        self.assertTrue(fresh.route_active(1))
        self.assertEqual(fresh.state(1)['posting_cursor'], 50)
        self.assertEqual(fresh.state(1)['auto_publish'], 0)
        self.assertEqual(fresh.queue_size(1), 0)

    async def test_history_is_indexed_without_publishing(self):
        async def history(source, after, head, limit):
            for entry in (item(1, 'Первый текст'), item(2, 'Первый текст')):
                if after < entry.id <= head:
                    yield entry

        telegram = SimpleNamespace(head=AsyncMock(return_value=2), history=history)
        self.assertFalse(await index_batch(self.archive, telegram, self.route.source))
        self.assertEqual(self.archive.indexed_count(1), 2)
        self.assertEqual(self.archive.queue_size(1), 0)
        self.assertEqual(self.archive.analyze(1, 'Первый текст')['overall'], 0)

    async def test_independent_modes_and_stop(self):
        telegram = SimpleNamespace(head=AsyncMock(return_value=10))
        control = self.control(telegram)
        await control.dispatch('/index off')
        await control.dispatch('/post on')
        self.assertFalse(self.archive.route_active(1))
        self.assertEqual(self.archive.state(1)['auto_publish'], 1)
        self.assertEqual(self.archive.state(1)['posting_cursor'], 10)
        await control.dispatch('/stop')
        self.assertEqual(self.archive.state(1)['auto_publish'], 0)
        self.assertFalse(self.archive.delivery_enabled(1))

    async def test_llm_switch_and_startup_failure_are_visible(self):
        control = self.control(SimpleNamespace())

        class FakeEncoder:
            identity = 'minilm-test'

            def encode(self, texts):
                return [[1, 0] for _ in texts]

        with patch('uniqueness.control.ensure_model'), patch('uniqueness.control.Encoder', return_value=FakeEncoder()):
            self.assertIn('включена', await control.dispatch('/llm on'))
            self.assertEqual(control.mode, 'minilm')
            self.archive.index(1, 1, '2025', 'Текст для повторной оценки.')
            semantic = await control.analyze('Текст для повторной оценки.')
            self.assertIn('выключена', await control.dispatch('/llm off'))
            classic = await control.analyze('Текст для повторной оценки.')
            self.assertIn('Новизна:', semantic)
            self.assertIn('Новизна:', classic)
        self.archive.set_setting('mode', 'minilm')
        self.archive.set_auto_publish(1, 1)
        restarted = self.control(SimpleNamespace(count=AsyncMock(return_value=0)))
        with patch('uniqueness.control.ensure_model', side_effect=PermissionError('model.onnx')):
            with self.assertRaises(PermissionError):
                await restarted.warm_model()
        self.assertEqual(self.archive.state(1)['auto_publish'], 0)
        self.assertIn('model.onnx', await restarted.status())

    async def test_new_post_enters_queue_and_report_replies_in_channel(self):
        async def history(source, after, head, limit):
            if after < 11 <= head:
                yield item(11, 'Новая публикация с текстом.')

        telegram = SimpleNamespace(head=AsyncMock(side_effect=[10, 11]), history=history)
        control = self.control(telegram)
        await control.dispatch('/post on')
        self.assertFalse(await control.watch(1))
        self.assertEqual(self.archive.queue_size(1), 1)

        transport = SimpleNamespace(forward=AsyncMock(return_value=41),
                                    send_report=AsyncMock(return_value=42))
        self.assertTrue(await deliver(self.archive, transport, self.route))
        transport.send_report.assert_awaited_once()
        self.assertEqual(transport.send_report.await_args.args[1], 41)
        self.assertEqual(self.archive.queue_size(1), 0)

        user = AsyncMock(return_value=SimpleNamespace(updates=[types.UpdateMessageID(id=42, random_id=99)]))
        api = Telegram(user, self.archive)
        self.assertEqual(await api.send_report(self.route, 41, 'Оценка', 99), 42)
        request = user.await_args.args[0]
        self.assertIsInstance(request, functions.messages.SendMessageRequest)
        self.assertEqual(request.peer, self.route.destination)
        self.assertEqual(request.reply_to.reply_to_msg_id, 41)

    async def test_an_repeats_and_responds_to_forward(self):
        self.archive.index(1, 1, '2025', 'Исходный текст для сравнения.')
        message = SimpleNamespace(message='Исходный текст для сравнения.')
        user = SimpleNamespace(get_messages=AsyncMock(return_value=message),
                               get_input_entity=AsyncMock(return_value=peer(1)))

        async def call(identity, operation, *args, **kwargs):
            return await operation(*args, **kwargs)

        control = self.control(SimpleNamespace(user=user, call=call))
        first = await control.analyze('https://t.me/source_channel/2')
        self.assertEqual(first, await control.analyze('https://t.me/source_channel/2'))
        self.assertIn('Новизна:', await control.analyze('Другой текст для сравнения.'))
        self.assertEqual(self.archive.queue_size(1), 0)

        await control.dispatch('/an', sender=100)
        event = SimpleNamespace(sender_id=100, is_private=True, raw_text='пересланный текст',
                                message=SimpleNamespace(fwd_from=SimpleNamespace(
                                    from_id=types.PeerChannel(1), channel_post=2)),
                                respond=AsyncMock())
        await control.handle(event)
        event.respond.assert_awaited_once()
        self.assertIn('Пост 2', event.respond.await_args.args[0])
        button = event.respond.await_args.kwargs['buttons'][0][0]
        callback = SimpleNamespace(sender_id=100, is_private=True, data=button.data,
                                   answer=AsyncMock())
        await control.handle_button(callback)
        self.assertEqual(self.archive.indexed_count(1), 2)
        self.assertEqual(self.archive.state(1)['cursor'], 1)
        self.assertEqual(self.archive.state(1)['scan_count'], 1)
        self.assertEqual(self.archive.queue_size(1), 0)
        await control.handle_button(callback)
        self.assertIn('Уже в базе', callback.answer.await_args.args[0])
        self.archive.index(1, 2, '2025', message.message)
        self.assertEqual(self.archive.indexed_count(1), 2)
        self.assertEqual(self.archive.state(1)['cursor'], 2)
        self.assertEqual(self.archive.queue_size(1), 0)

    async def test_manual_text_is_indexed_without_background_progress(self):
        control = self.control(SimpleNamespace())
        event = SimpleNamespace(sender_id=100, is_private=True, raw_text='/an Текст вручную.',
                                message=SimpleNamespace(fwd_from=None), respond=AsyncMock())
        await control.handle(event)
        button = event.respond.await_args.kwargs['buttons'][0][0]
        callback = SimpleNamespace(sender_id=100, is_private=True, data=button.data,
                                   answer=AsyncMock())
        await control.handle_button(callback)
        self.assertEqual(self.archive.indexed_count(1), 1)
        self.assertEqual(self.archive.state(1)['cursor'], 0)
        self.assertEqual(self.archive.state(1)['scan_count'], 0)
        self.assertEqual(self.archive.queue_size(1), 0)
        self.assertIn('Новизна:', await control.analyze('Текст вручную.'))

    async def test_link_checks_whole_channel_except_itself(self):
        self.archive.index(1, 1, '2025', 'Повторяемый тезис про мужчин.')
        self.archive.index(1, 2, '2025', 'Совсем другая публикация.')
        self.archive.index(1, 3, '2025', 'Повторяемый тезис про мужчин.')
        self.archive.set_setting('route_name:1', '@source_channel → @my_channel')
        user = SimpleNamespace(get_messages=AsyncMock(return_value=SimpleNamespace(
            message='Повторяемый тезис про мужчин.')),
            get_input_entity=AsyncMock(return_value=peer(1)))

        async def call(identity, operation, *args, **kwargs):
            return await operation(*args, **kwargs)

        control = self.control(SimpleNamespace(user=user, call=call))
        answer = await control.analyze('https://t.me/source_channel/1')
        self.assertIn('Новизна: 0%', answer)
        self.assertIn('https://t.me/source_channel/3', answer)
        self.assertIn('Новизна: 0%', await control.analyze('Повторяемый тезис про мужчин.'))

    async def test_channel_frequency_and_semantic_coverage(self):
        for post in range(1, 12):
            self.archive.index(1, post, '2025', f'Мужчины снова повторяют знакомый тезис номер {post}.')
        repeated = self.archive.analyze(1, 'Мужчины снова повторяют знакомый тезис.')
        technical = self.archive.analyze(1, 'Документация компилятора описывает асинхронный протокол.')
        self.assertLess(repeated['words'], technical['words'])
        self.assertLess(repeated['phrases'], technical['phrases'])
        self.assertLess(repeated['overall'], technical['overall'])

        class FakeEncoder:
            identity = 'semantic-test'

            def encode(self, texts):
                return [[1, 0] if 'мужчин' in text else [0, 1] for text in texts]

        self.archive.set_encoder(FakeEncoder())
        self.archive.rebuild_index(1)
        self.archive.index(1, 1, '2025', 'Мужчины повторяют одну мысль.')
        self.archive.index(1, 2, '2025', 'Текст документации другой области.')
        self.assertEqual(self.archive.analyze(1, 'Мужчины снова говорят похожее.')['theme'], 0)
        self.assertEqual(self.archive.analyze(1, 'Совершенно иной материал.', exclude=2)['theme'], 100)

    async def test_llm_backfills_posts_indexed_while_off(self):
        self.archive.index(1, 1, '2025', 'Мужчины повторяют старую мысль.')

        class FakeEncoder:
            identity = 'backfill-test'

            def encode(self, texts):
                return [[1, 0] for _ in texts]

        self.archive.set_encoder(FakeEncoder())
        self.assertTrue(self.archive.analyze(1, 'Новая формулировка')['model_pending'])
        self.assertTrue(self.archive.fill_vectors(1))
        self.assertEqual(self.archive.vector_count(1), 1)
        self.assertFalse(self.archive.analyze(1, 'Новая формулировка')['model_pending'])
        self.assertEqual(self.archive.state(1)['cursor'], 1)
        self.assertEqual(self.archive.queue_size(1), 0)

    async def test_posts_with_only_emoji_do_not_block_vector_backfill(self):
        self.archive.index(1, 1, '2025', '🧠🔥')
        self.archive.index(1, 2, '2025', 'Текст с эмодзи 🧠')
        self.assertEqual(self.archive.indexed_count(1), 1)
        self.assertEqual(self.archive.state(1)['scan_count'], 2)
        with self.assertRaisesRegex(ValueError, 'нет слов'):
            self.archive.add_reference(1, None, '🧠🔥')

        class FakeEncoder:
            identity = 'emoji-test'

            def encode(self, texts):
                return [[1, 0] for _ in texts]

        self.archive.set_encoder(FakeEncoder())
        self.assertTrue(self.archive.fill_vectors(1))
        self.assertFalse(self.archive.fill_vectors(1))
        self.assertEqual(self.archive.vector_count(1), 1)
        self.assertFalse(self.archive.analyze(1, 'Другой текст')['model_pending'])

    async def test_old_wordless_posts_are_removed_on_open(self):
        path = Path(self.temp.name) / 'old-wordless.sqlite3'
        old = Archive(path)
        with old.db:
            old.db.execute('INSERT INTO posts VALUES (?, ?, ?, ?, 0, 0, 0, 0)',
                           (1, 1, '🧠🔥', b'x'))
        old.close()
        reopened = Archive(path)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.indexed_count(1), 0)

    async def test_rebuild_index_resets_progress_without_touching_publication(self):
        self.archive.index(1, 1, '2025', 'Исходный текст.')
        self.archive.enqueue(1, 5, 'Оценка')
        self.archive.set_auto_publish(1, 1)
        control = self.control(SimpleNamespace())
        self.assertIn('началась заново', await control.dispatch('/index rebuild'))
        self.assertEqual(self.archive.indexed_count(1), 0)
        self.assertEqual(self.archive.state(1)['cursor'], 0)
        self.assertEqual(self.archive.state(1)['scan_count'], 0)
        self.assertTrue(self.archive.route_active(1))
        self.assertEqual(self.archive.state(1)['auto_publish'], 1)
        self.assertEqual(self.archive.queue_size(1), 1)

    async def test_manual_post_button_publishes_text_and_report(self):
        telegram = SimpleNamespace(send_text=AsyncMock(return_value=50),
                                   send_report=AsyncMock(return_value=51))
        control = self.control(telegram)
        event = SimpleNamespace(sender_id=100, is_private=True, raw_text='/an Проверяемый текст.',
                                message=SimpleNamespace(fwd_from=None), respond=AsyncMock())
        await control.handle(event)
        button = event.respond.await_args.kwargs['buttons'][0][1]
        callback = SimpleNamespace(sender_id=100, is_private=True, data=button.data,
                                   answer=AsyncMock())
        await control.handle_button(callback)
        telegram.send_text.assert_awaited_once()
        telegram.send_report.assert_awaited_once()
        self.assertEqual(telegram.send_report.await_args.args[1], 50)
        self.assertEqual(self.archive.queue_size(1), 0)
        await control.handle_button(callback)
        self.assertIn('Уже опубликовано', callback.answer.await_args.args[0])

    async def test_queue_pause_and_clear(self):
        self.archive.enqueue(1, 5, 'Оценка')
        control = self.control(SimpleNamespace(head=AsyncMock(return_value=5)))
        self.assertIn('1 ожидают', await control.dispatch('/queue'))
        await control.dispatch('/post on')
        await control.dispatch('/queue pause')
        self.assertFalse(self.archive.delivery_enabled(1))
        self.assertIn('1', await control.dispatch('/queue clear'))
        self.assertEqual(self.archive.queue_size(1), 0)

    async def test_clear_during_forward_prevents_report(self):
        self.archive.enqueue(1, 7, 'Оценка')

        async def forward(*args, **kwargs):
            self.archive.clear_queue()
            return 70

        transport = SimpleNamespace(forward=forward, send_report=AsyncMock())
        self.assertFalse(await deliver(self.archive, transport, self.route))
        transport.send_report.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
