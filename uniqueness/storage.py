import json
import secrets
import sqlite3
from contextlib import closing
from pathlib import Path

from .analysis import compare, deserialize, extract, index_groups, prepare_for_scoring, serialize

SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS channels (
    source INTEGER PRIMARY KEY,
    destination INTEGER NOT NULL,
    peers TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    auto_publish INTEGER NOT NULL DEFAULT 0,
    delivery_enabled INTEGER NOT NULL DEFAULT 0,
    cursor INTEGER NOT NULL DEFAULT 0,
    snapshot INTEGER,
    history_done INTEGER NOT NULL DEFAULT 0,
    scan_count INTEGER NOT NULL DEFAULT 0,
    posting_cursor INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS posts (
    source INTEGER NOT NULL,
    id INTEGER NOT NULL,
    original TEXT NOT NULL,
    features BLOB NOT NULL,
    wording INTEGER NOT NULL,
    topic INTEGER NOT NULL,
    meaning INTEGER NOT NULL,
    style INTEGER NOT NULL,
    PRIMARY KEY (source, id)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS postings (
    source INTEGER NOT NULL,
    metric INTEGER NOT NULL,
    key INTEGER NOT NULL,
    post INTEGER NOT NULL,
    PRIMARY KEY (source, metric, key, post)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS outbox (
    source INTEGER NOT NULL,
    post INTEGER NOT NULL,
    result TEXT NOT NULL,
    forward_random INTEGER NOT NULL,
    report_random INTEGER NOT NULL,
    forward_id INTEGER,
    report_id INTEGER,
    skipped TEXT,
    PRIMARY KEY (source, post)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS outbox_pending ON outbox(source, post)
    WHERE report_id IS NULL AND skipped IS NULL;
CREATE TABLE IF NOT EXISTS copied_messages (
    source INTEGER NOT NULL,
    destination INTEGER NOT NULL,
    post INTEGER NOT NULL,
    published_id INTEGER NOT NULL,
    PRIMARY KEY (source, destination, post)
) WITHOUT ROWID;
"""


class Archive:
    def __init__(self, path: Path, analyzer_id='classic-v1', encoder=None):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;')
        existing = self.db.execute("SELECT 1 FROM sqlite_master WHERE name='channels'").fetchone()
        if existing and 'peers' not in {row['name'] for row in self.db.execute('PRAGMA table_info(channels)')}:
            self.db.close()
            raise ValueError('Старая схема базы. Останови контейнер и очисти archive.sqlite3 перед запуском cgr.')
        self.db.executescript(SCHEMA)
        self.encoder = encoder
        self.db.execute('INSERT OR IGNORE INTO metadata VALUES (?, ?)', ('analyzer', analyzer_id))
        self.db.commit()

    def close(self):
        self.db.close()

    def pin(self, key, value):
        value = str(value)
        with self.db as db:
            db.execute('INSERT OR IGNORE INTO metadata VALUES (?, ?)', (key, value))
            if self.setting(key) != value:
                raise ValueError(f'Настройка {key} отличается от сохранённой в базе')

    def setting(self, key, default=None):
        row = self.db.execute('SELECT value FROM metadata WHERE key = ?', (key,)).fetchone()
        return row[0] if row else default

    def set_setting(self, key, value):
        with self.db as db:
            db.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', (key, str(value)))

    def set_encoder(self, encoder):
        if encoder is not None:
            self.pin('semantic_encoder', encoder.identity)
        self.encoder = encoder

    def route_records(self):
        for row in self.db.execute('SELECT source, peers, active FROM channels'):
            yield row['source'], json.loads(row['peers']), row['active']

    def state(self, source):
        return self.db.execute('SELECT * FROM channels WHERE source = ?', (source,)).fetchone()

    def save_route(self, source, destination, peers, *, active=1):
        old = self.state(source)
        with self.db as db:
            if old and old['destination'] != destination:
                db.execute('DELETE FROM outbox WHERE source = ? AND report_id IS NULL', (source,))
            db.execute('INSERT INTO channels (source, destination, peers, active) VALUES (?, ?, ?, ?) '
                       'ON CONFLICT(source) DO UPDATE SET destination=excluded.destination, peers=excluded.peers',
                       (source, destination, json.dumps(peers), active))

    def route_active(self, source):
        row = self.state(source)
        return bool(row and row['active'])

    def set_route_active(self, source, active):
        with self.db as db:
            db.execute('UPDATE channels SET active = ? WHERE source = ?', (int(active), source))

    def set_auto_publish(self, source, enabled):
        with self.db as db:
            db.execute('UPDATE channels SET auto_publish = ? WHERE source = ?', (int(enabled), source))

    def delivery_enabled(self, source):
        row = self.state(source)
        return bool(row and row['delivery_enabled'])

    def set_delivery_enabled(self, source, enabled):
        with self.db as db:
            db.execute('UPDATE channels SET delivery_enabled = ? WHERE source = ?', (int(enabled), source))

    def set_posting_cursor(self, source, post):
        with self.db as db:
            db.execute('UPDATE channels SET posting_cursor = MAX(posting_cursor, ?) WHERE source = ?', (post, source))

    def stop_all(self):
        with self.db as db:
            db.execute('UPDATE channels SET active = 0, auto_publish = 0, delivery_enabled = 0')

    def snapshot(self, source, head):
        with self.db as db:
            db.execute('UPDATE channels SET snapshot = ? WHERE source = ? AND snapshot IS NULL', (head, source))

    def finish_snapshot(self, source, head):
        with self.db as db:
            db.execute('UPDATE channels SET cursor = MAX(cursor, ?), snapshot = NULL, history_done = 1 '
                       'WHERE source = ? AND snapshot = ?', (head, source, head))

    def index(self, source, post, published, original):
        if self.state(source)['cursor'] >= post:
            return
        with self.db as db:
            if original and original.strip():
                self._store_post(source, post, original)
            db.execute('UPDATE channels SET cursor = ?, scan_count = scan_count + 1 WHERE source = ?',
                       (post, source))

    def add_reference(self, source, post, original):
        if self.state(source) is None:
            raise ValueError('Канал не настроен')
        if not original or not original.strip():
            raise ValueError('У поста нет текста для индексации')
        with self.db:
            if post is None:
                row = self.db.execute('SELECT MIN(id) FROM posts WHERE source = ? AND id < 0',
                                      (source,)).fetchone()
                post = min(0, row[0] or 0) - 1
            return self._store_post(source, post, original)

    def _store_post(self, source, post, original):
        if self.db.execute('SELECT 1 FROM posts WHERE source = ? AND id = ?', (source, post)).fetchone():
            return False
        features = extract(original, self.encoder)
        packed = serialize(features)
        keys = {(metric, key) for metric, group in index_groups(features) for key in group}
        probes = list(index_groups(features, query=True))
        scores = self.score(source, post if post > 0 else 2**63 - 1,
                            prepare_for_scoring(features), probes)
        self.db.execute('INSERT INTO posts VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                        (source, post, original, packed, *scores))
        self.db.executemany('INSERT OR IGNORE INTO postings VALUES (?, ?, ?, ?)',
                            ((source, metric, key, post) for metric, key in keys))
        return True

    def score(self, source, post, current, probes):
        similarities = [0.0] * 4
        for candidate in self.candidates(source, post, probes):
            row = self.db.execute('SELECT original, features FROM posts WHERE source = ? AND id = ?',
                                  (source, candidate)).fetchone()
            previous = deserialize(row['features'])
            if 'vectors' in current and 'vectors' not in previous and self.encoder:
                previous = prepare_for_scoring(extract(row['original'], self.encoder))
            compare(current, previous, similarities)
            if min(similarities) >= 1:
                break
        return tuple(round(100 * (1 - min(1.0, value))) for value in similarities)

    def candidates(self, source, post, probes):
        votes = {}
        for metric, keys in probes:
            for key in keys:
                rows = self.db.execute('SELECT post FROM postings WHERE source = ? AND metric = ? AND key = ? '
                                       'AND post < ? ORDER BY post DESC LIMIT 25',
                                       (source, metric, key, post))
                for row in rows:
                    votes[row['post']] = votes.get(row['post'], 0) + 1
        selected = sorted(votes, key=lambda candidate: (votes[candidate], candidate), reverse=True)[:25]
        recent = self.db.execute('SELECT id FROM posts WHERE source = ? AND id < ? ORDER BY id DESC LIMIT 8',
                                 (source, post))
        yield from dict.fromkeys([*selected, *(row['id'] for row in recent)])

    def indexed_count(self, source):
        return self.db.execute('SELECT count(*) FROM posts WHERE source = ?', (source,)).fetchone()[0]

    def enqueue(self, source, post, result):
        with self.db as db:
            db.execute('INSERT OR IGNORE INTO outbox (source, post, result, forward_random, report_random) '
                       'VALUES (?, ?, ?, ?, ?)', (source, post, result,
                                                  secrets.randbits(63) or 1, secrets.randbits(63) or 1))

    def next_job(self, source):
        return self.db.execute('SELECT * FROM outbox WHERE source = ? AND report_id IS NULL '
                               'AND skipped IS NULL ORDER BY post LIMIT 1', (source,)).fetchone()

    def job_pending(self, source, post):
        return self.db.execute('SELECT 1 FROM outbox WHERE source = ? AND post = ? '
                               'AND report_id IS NULL AND skipped IS NULL', (source, post)).fetchone() is not None

    def update_job(self, source, post, *, forward=None, result=None, skipped=None):
        with self.db as db:
            db.execute('UPDATE outbox SET forward_id = COALESCE(?, forward_id), '
                       'report_id = COALESCE(?, report_id), skipped = COALESCE(?, skipped) '
                       'WHERE source = ? AND post = ?', (forward, result, skipped, source, post))

    def queue_size(self, source):
        return self.db.execute('SELECT count(*) FROM outbox WHERE source = ? AND report_id IS NULL '
                               'AND skipped IS NULL', (source,)).fetchone()[0]

    def clear_queue(self):
        with self.db as db:
            return db.execute('DELETE FROM outbox WHERE report_id IS NULL').rowcount

    def copied_id(self, source, destination, post):
        row = self.db.execute('SELECT published_id FROM copied_messages WHERE source = ? '
                              'AND destination = ? AND post = ?', (source, destination, post)).fetchone()
        return row[0] if row else None

    def store_copies(self, source, destination, copies):
        with self.db as db:
            db.executemany('INSERT OR IGNORE INTO copied_messages VALUES (?, ?, ?, ?)',
                           ((source, destination, post, published) for post, published in copies.items()))

    def defer_flood(self, identity, until):
        self.set_setting(f'flood:{identity}', until)

    def flood_until(self, identity):
        return float(self.setting(f'flood:{identity}', 0))

    def remember_error(self, source, message):
        self.set_setting(f'error:{source}', message[:1000])

    def error(self, source):
        return self.setting(f'error:{source}')


def read_status(path):
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute('SELECT source, destination, active, auto_publish, '
                                                'delivery_enabled, scan_count, posting_cursor FROM channels')]
