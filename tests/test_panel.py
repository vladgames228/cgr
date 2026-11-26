import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from telethon import types

from uniqueness.control import Control
from uniqueness.storage import Archive
from uniqueness.worker import Incoming, Route, deliver_manual, index_batch


def route():
    return Route(*(types.InputPeerChannel(identifier, 11) for identifier in (1, 2, 3, 3)))


class PanelTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.archive = Archive(Path(self.temp.name) / 'archive.sqlite3')
        self.addCleanup(self.archive.close)
        self.route = route()
        self.archive.save_route(1, 2, 3, [(i, 11) for i in (1, 2, 3, 3)], active=0)

    async def test_indexing_does_not_publish_and_limit_pauses(self):
        self.archive.set_index_limit(1, 2)
        self.archive.set_route_active(1, 1)
        items = [Incoming(i, datetime.now(timezone.utc), f'Текст {i}') for i in range(1, 4)]

        async def history(source, after, head, limit):
            for item in items:
                if after < item.id <= head:
                    yield item

        telegram = SimpleNamespace(head=AsyncMock(return_value=3), history=history)
        self.assertFalse(await index_batch(self.archive, telegram, self.route.source))
        self.assertEqual(self.archive.state(1)['scan_count'], 2)
        self.assertEqual(self.archive.queue_size(1), 0)
        self.assertEqual(self.archive.db.execute('SELECT active FROM route_settings').fetchone()[0], 0)

    async def test_same_post_can_be_sent_twice_with_distinct_reports(self):
        self.archive.index(1, 5, '2025-01-01', 'Первый текст для анализа.')
        self.assertIsNone(self.archive.next_job(1))
        self.archive.add_manual(100, 10, 1, 5, 'Оценка A')
        self.archive.add_manual(100, 11, 1, 5, 'Оценка B')
        telegram = SimpleNamespace(
            forward=AsyncMock(side_effect=[51, 52]),
            discussion_root=AsyncMock(side_effect=[71, 72]),
            send_report=AsyncMock(side_effect=[91, 92]),
        )
        self.assertTrue(await deliver_manual(self.archive, telegram, self.route))
        self.assertTrue(await deliver_manual(self.archive, telegram, self.route))
        self.assertEqual(self.archive.queue_size(1), 0)
        self.assertEqual([call.args[2] for call in telegram.send_report.await_args_list],
                         ['Оценка A', 'Оценка B'])
        self.assertTrue(all(call.kwargs['reuse_copy'] is False
                            for call in telegram.forward.await_args_list))

    async def test_reanalysis_after_mode_changes(self):
        self.archive.index(1, 1, '2025-01-01', 'Дорогу закрыли из-за сильного снегопада.')
        item = SimpleNamespace(message='Трассу перекрыли, потому что выпало много снега.')
        telegram = SimpleNamespace(user=SimpleNamespace(get_messages=AsyncMock(return_value=item)),
                                   call=lambda identity, operation, *args, **kwargs: operation(*args, **kwargs))
        control = Control(self.archive, telegram, self.temp.name, {100})
        classic = await control.analyze_post('-1001', 2)

        class FakeEncoder:
            identity = 'minilm-test'

            def encode(self, texts):
                return [[1, 0] for _ in texts]

        with patch('uniqueness.control.ensure_model'), patch('uniqueness.control.Encoder', return_value=FakeEncoder()):
            await control.set_mode('minilm')
            semantic = await control.analyze_post('-1001', 2)
            await control.set_mode('classic')
            self.assertEqual(await control.analyze_post('-1001', 2), classic)
            await control.set_mode('minilm')
            self.assertEqual(await control.analyze_post('-1001', 2), semantic)
        self.assertNotEqual(classic, semantic)

    async def test_button_dialog_configures_route_and_accepts_custom_limit(self):
        telegram = SimpleNamespace(configure=AsyncMock(return_value=self.route),
                                   count=AsyncMock(return_value=321))
        control = Control(self.archive, telegram, self.temp.name, {100})

        def event(data=None, text=''):
            return SimpleNamespace(sender_id=100, is_private=True, data=data, raw_text=text,
                                   message=SimpleNamespace(fwd_from=None),
                                   answer=AsyncMock(), respond=AsyncMock())

        choice = event(b'ask:route')
        await control.handle_button(choice)
        choice.answer.assert_awaited_once()
        reply = event(text='@source_channel @destination_channel')
        await control.handle(reply)
        self.assertIn('📣 -1001', reply.respond.await_args.args[0])
        self.assertEqual(control.routes[1][1], 0)

        choice = event(b'ask:index:1')
        await control.handle_button(choice)
        reply = event(text='25')
        await control.handle(reply)
        self.assertEqual(self.archive.state(1)['index_limit'], 25)
        self.assertEqual(control.routes[1][1], 1)
