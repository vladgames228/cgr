import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from telethon import errors, functions, types

from uniqueness.analysis import compare, extract, prepare_for_scoring
from uniqueness.storage import Archive, read_status
from uniqueness.telegram import RetryLater, Telegram, channel_reference, sent_id
from uniqueness.worker import Incoming, Route, Undeliverable, deliver, index_batch


def input_peer(identifier):
    return types.InputPeerChannel(identifier, 11)


def fake_incoming(identifier, text):
    return Incoming(identifier, datetime(2025, 1, 1, tzinfo=timezone.utc), text)


class Tests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'archive.sqlite3'
        self.db = Archive(self.path)
        self.addCleanup(lambda: self.db.close())
        self.db.register(1, 2, 3)
        self.route = Route(input_peer(1), input_peer(2), input_peer(3), input_peer(3))

    def reopen(self):
        self.db.close()
        self.db = Archive(self.path)

    def scores(self, source, identifier):
        row = self.db.db.execute(
            'SELECT wording, topic, meaning, style FROM posts WHERE source = ? AND id = ?',
            (source, identifier)).fetchone()
        return tuple(row)

    def add(self, identifier, text, source=1):
        self.db.index(source, identifier, '2025-01-01T00:00:00+00:00', text)

    def test_only_prior_posts_and_isolated_sources(self):
        self.db.register(4, 2, 3)
        self.add(1, 'Подпись к фотографии автомобиля.')
        self.add(2, None)
        self.add(3, 'Подпись к фотографии автомобиля.')
        self.add(9, 'Подпись к фотографии автомобиля.', source=4)
        self.add(3, 'Отредактированный текст.')
        self.assertEqual(self.scores(1, 1), (100, 100, 100, 100))
        self.assertEqual(self.scores(4, 9), (100, 100, 100, 100))
        self.assertEqual(self.scores(1, 3), (0, 0, 0, 0))
        self.assertEqual(self.db.state(1)['cursor'], 3)
        self.assertEqual(read_status(self.path)[0]['indexed'], 2)
        original = self.db.db.execute('SELECT original FROM posts WHERE source = 1 AND id = 3').fetchone()[0]
        self.assertEqual(original, 'Подпись к фотографии автомобиля.')

    def test_distant_repeated_fragment(self):
        phrase = 'В бухте установили автономные датчики температуры и солёности воды.'
        self.add(1, f'Открылась выставка редких карт. {phrase} Работает парк.')
        for number in range(2, 48):
            self.add(number, f'Зал концертной площадки принимает гостей. Билет номер {number}.')
        self.add(48, f'Начался ремонт трамвайного депо. {phrase} Обновление займёт квартал.')
        wording, topic, semantic, _ = self.scores(1, 48)
        self.assertLessEqual(max(wording, topic, semantic), 10)

    def test_synonyms_negation_and_characters(self):
        self.add(1, 'Компания приобрела автомобиль для сотрудника.')
        self.add(2, 'Фирма купила машину для работника.')
        self.add(3, 'Компания не приобрела автомобиль для сотрудника.')
        wording, theme, semantic, _ = self.scores(1, 2)
        self.assertLess(semantic, min(wording, theme))
        self.assertGreater(self.scores(1, 3)[2], semantic)
        self.add(4, '🧩🧩🧩🧩🧩🧩')
        self.add(5, '🧩🧩🧩🧩🧩🧩')
        self.assertEqual(self.scores(1, 5)[0], 0)

    async def test_resume_history_and_follow_new_posts(self):
        transport = SimpleNamespace(head=AsyncMock(side_effect=[9, 12]), history=None)
        seen = []

        async def history(source, after, head, limit):
            seen.append((after, head, limit))
            items = [fake_incoming(2, 'Подпись'), fake_incoming(5, None),
                     fake_incoming(9, 'Текст'), fake_incoming(12, 'Новый текст')]
            number = 0
            for item in items:
                if after < item.id <= head and number < limit:
                    if len(seen) == 1 and item.id == 5:
                        raise ConnectionError('connection lost')
                    yield item
                    number += 1

        transport.history = history
        with self.assertRaises(ConnectionError):
            await index_batch(self.db, transport, self.route.source, 2)
        self.reopen()
        self.assertEqual(self.db.state(1)['cursor'], 2)
        self.assertEqual(self.db.state(1)['snapshot'], 9)
        self.assertFalse(await index_batch(self.db, transport, self.route.source, 2))
        self.assertEqual(self.db.state(1)['history_done'], 1)
        self.assertFalse(await index_batch(self.db, transport, self.route.source, 2))
        self.assertEqual(seen, [(0, 9, 2), (2, 9, 2), (9, 12, 2)])
        self.assertEqual(read_status(self.path)[0]['indexed'], 3)

    async def test_delivery_reuses_ids_after_lost_responses(self):
        self.add(5, 'Подпись к фото.')
        self.add(8, 'Следующая публикация.')
        forward_calls, report_calls = [], []

        async def forward(route, post, random_id):
            forward_calls.append((post, random_id))
            if len(forward_calls) == 1:
                raise ConnectionError('forward accepted, reply lost')
            return 101

        async def report(route, root, text, random_id):
            report_calls.append((root, text, random_id))
            if len(report_calls) == 1:
                raise ConnectionError('comment accepted, reply lost')
            return 701

        transport = SimpleNamespace(forward=forward, discussion_root=AsyncMock(return_value=501),
                                    send_report=report)
        with self.assertRaises(ConnectionError):
            await deliver(self.db, transport, self.route)
        self.reopen()
        with self.assertRaises(ConnectionError):
            await deliver(self.db, transport, self.route)
        self.reopen()
        self.assertTrue(await deliver(self.db, transport, self.route))
        self.assertEqual(forward_calls[0], forward_calls[1])
        self.assertEqual(report_calls[0], report_calls[1])
        transport.discussion_root.assert_awaited_once_with(self.route, 101)
        self.assertEqual(report_calls[0][1].splitlines(), [
            'Уникальность формулировок: 100%',
            'Уникальность тематики: 100%',
            'Уникальность смысла: 100%',
            'Уникальность стиля: 100%',
        ])
        self.assertEqual(self.db.next_job(1)['id'], 8)

    async def test_protected_forward_keeps_analysis(self):
        self.add(2, 'Содержимое защищенной публикации.')
        transport = SimpleNamespace(forward=AsyncMock(side_effect=Undeliverable('protected')))
        with self.assertLogs('uniqueness.worker', 'WARNING'):
            self.assertTrue(await deliver(self.db, transport, self.route))
        self.assertIsNone(self.db.next_job(1))
        self.assertEqual(read_status(self.path)[0]['skipped'], 1)
        self.assertEqual(self.scores(1, 2), (100, 100, 100, 100))

    async def test_flood_wait_survives_restart(self):
        api = Telegram(None, None, self.db)
        rejected = AsyncMock(side_effect=errors.FloodWaitError(request=None, capture=27))
        with self.assertRaises(RetryLater):
            await api.call('user', rejected)
        self.reopen()
        api = Telegram(None, None, self.db)
        with self.assertRaises(RetryLater):
            await api.call('user', rejected)
        rejected.assert_awaited_once()
        self.assertGreater(api.pause_remaining('user'), 25)
        self.assertEqual(api.pause_remaining('bot'), 0)

    async def test_mtproto_request_targets_and_reply(self):
        forward_updates = SimpleNamespace(updates=[types.UpdateMessageID(id=87, random_id=10)])
        root = SimpleNamespace(id=157, peer_id=types.PeerChannel(3))
        user = AsyncMock(side_effect=[forward_updates, SimpleNamespace(messages=[root])])
        bot = AsyncMock(return_value=SimpleNamespace(updates=[types.UpdateMessageID(id=601, random_id=20)]))
        api = Telegram(user, bot, self.db)
        self.assertEqual(await api.forward(self.route, 77, 10), 87)
        self.assertEqual(await api.discussion_root(self.route, 87), 157)
        self.assertEqual(await api.send_report(self.route, 157, 'four lines', 20), 601)
        forward, discussion = [call.args[0] for call in user.await_args_list]
        self.assertIsInstance(forward, functions.messages.ForwardMessagesRequest)
        self.assertEqual(forward.id, [77])
        self.assertEqual(forward.random_id, [10])
        self.assertEqual(forward.to_peer, self.route.destination)
        self.assertEqual(discussion.msg_id, 87)
        send = bot.await_args.args[0]
        self.assertEqual(send.peer, self.route.bot_discussion)
        self.assertEqual(send.reply_to.reply_to_msg_id, 157)
        self.assertEqual(send.random_id, 20)
        with self.assertRaises(RetryLater):
            sent_id(SimpleNamespace(updates=[]), 10, self.route.destination)

    async def test_deleted_comment_thread_does_not_block_queue(self):
        self.add(5, 'Первая публикация.')
        self.add(6, 'Вторая публикация.')
        self.db.update_delivery(1, 5, forward=101, root=501)
        user = SimpleNamespace(get_messages=AsyncMock(return_value=None))
        bot = AsyncMock(side_effect=errors.BadRequestError(None, 'TOPIC_ID_INVALID', 400))
        with self.assertLogs('uniqueness.worker', 'WARNING'):
            self.assertTrue(await deliver(self.db, Telegram(user, bot, self.db), self.route))
        self.assertEqual(self.db.next_job(1)['id'], 6)
        self.assertEqual(read_status(self.path)[0]['skipped'], 1)

    def test_reject_route_and_feature_identity_changes(self):
        self.assertEqual(channel_reference('https://t.me/s/channel_foo/', public=True), '@channel_foo')
        self.assertEqual(channel_reference('-1001234'), -1001234)
        for invalid in ('https://t.me/foo_bar/3', 'https://t.me/+abcde', 'https://example.com/channel_foo'):
            with self.assertRaises(ValueError):
                channel_reference(invalid, public=True)
        with self.assertRaises(ValueError):
            self.db.register(1, 4, 3)
        with self.assertRaises(ValueError):
            self.db.pin('analyzer', 'another-model')


@unittest.skipUnless(os.getenv('TEST_MODEL_DIR'), 'TEST_MODEL_DIR is not set')
class EncoderTest(unittest.TestCase):
    def test_real_model_ranks_deep_paraphrase_above_unrelated_text(self):
        from uniqueness.encoder import Encoder

        model = Encoder(os.environ['TEST_MODEL_DIR'])
        source = 'Дорогу закрыли из-за сильного снегопада.'
        paraphrase = 'Трассу перекрыли, потому что выпало много снега.'
        unrelated = 'В океанариуме открылась выставка тропических рыб.'

        def similarity(left, right, encoder=None):
            current = prepare_for_scoring(extract(left, encoder))
            previous = prepare_for_scoring(extract(right, encoder))
            scores = [0.0] * 4
            compare(current, previous, scores)
            return scores[2]

        baseline = similarity(source, paraphrase)
        improved = similarity(source, paraphrase, model)
        self.assertGreater(improved, baseline + 0.2)
        self.assertGreater(improved, similarity(source, unrelated, model) + 0.2)
