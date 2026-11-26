import json
import math
import os
import secrets
import sqlite3
from contextlib import closing
from pathlib import Path

from .analysis import content, corpus_counts, deserialize, extract, novelty, phrases, semantic, serialize, words

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
        self._corpus_cache = {}
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

    def rebuild_index(self, source):
        with self.db as db:
            db.execute('DELETE FROM postings WHERE source = ?', (source,))
            db.execute('DELETE FROM posts WHERE source = ?', (source,))
            db.execute('UPDATE channels SET cursor = 0, snapshot = NULL, history_done = 0, '
                       'scan_count = 0, active = 1 WHERE source = ?', (source,))
        self._corpus_cache.pop(source, None)

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
        packed = serialize(extract(original, self.encoder))
        self.db.execute('INSERT INTO posts VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                        (source, post, original, packed, 0, 0, 0, 0))
        self._corpus_cache.pop(source, None)
        return True

    def fill_vectors(self, source, batch_size=4):
        if self.encoder is None:
            return False
        missing = []
        for row in self.db.execute('SELECT id, original, features FROM posts WHERE source = ?', (source,)):
            if not deserialize(row['features']).get('vectors'):
                missing.append(row)
                if len(missing) == batch_size:
                    break
        if not missing:
            return False
        with self.db as db:
            for row in missing:
                db.execute('UPDATE posts SET features = ? WHERE source = ? AND id = ?',
                           (serialize(extract(row['original'], self.encoder)), source, row['id']))
        self._corpus_cache.pop(source, None)
        return True

    def _corpus(self, source):
        if source not in self._corpus_cache:
            rows = list(self.db.execute('SELECT id, original, features FROM posts WHERE source = ?', (source,)))
            self._corpus_cache[source] = (rows, *corpus_counts(rows))
        return self._corpus_cache[source]

    def analyze(self, source, text, exclude=None):
        rows, term_counts, phrase_counts, per_post = self._corpus(source)
        others = len(rows) - (exclude in per_post)
        terms = content(text)
        combinations = phrases(terms)
        owned_terms, owned_phrases = per_post.get(exclude, (set(), set()))
        if not others:
            return {'overall': 100, 'words': 100, 'phrases': 100, 'theme': None,
                    'closest': None, 'model_pending': False}
        normalized = words(text)
        for row in rows:
            if row['id'] != exclude and words(row['original']) == normalized:
                return {'overall': 0, 'words': 0, 'phrases': 0,
                        'theme': 0 if self.encoder else None, 'closest': row['id'],
                        'model_pending': False}
        vocabulary = novelty(terms, term_counts, owned_terms, others, max(2, math.ceil(others * 0.02)))
        expressions = novelty(combinations, phrase_counts, owned_phrases, others, 1)
        theme, closest, pending = None, None, False
        if self.encoder:
            pending = any(not deserialize(row['features']).get('vectors') for row in rows if row['id'] != exclude)
            if not pending:
                theme, closest = semantic(extract(text, self.encoder)['vectors'], rows, exclude)
        if closest is None and terms:
            query = set(terms)
            closest = max((row['id'] for row in rows if row['id'] != exclude),
                          key=lambda post: len(query & per_post[post][0]), default=None)
            if closest is not None and not (query & per_post[closest][0]):
                closest = None
        measures = [vocabulary, expressions]
        if theme is not None:
            measures.append(theme)
        overall = min(measures)
        return {'overall': overall, 'words': vocabulary, 'phrases': expressions,
                'theme': theme, 'closest': closest, 'model_pending': pending}

    def indexed_count(self, source):
        return self.db.execute('SELECT count(*) FROM posts WHERE source = ?', (source,)).fetchone()[0]

    def vector_count(self, source):
        return sum(bool(deserialize(row['features']).get('vectors')) for row in
                   self.db.execute('SELECT features FROM posts WHERE source = ?', (source,)))

    def size_mb(self):
        path = self.db.execute('PRAGMA database_list').fetchone()['file']
        return sum(os.path.getsize(path + suffix) for suffix in ('', '-wal', '-shm')
                   if os.path.exists(path + suffix)) / (1024 * 1024)

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
