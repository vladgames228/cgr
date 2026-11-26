import json
import secrets
import sqlite3
import zlib
from contextlib import closing
from pathlib import Path

from .analysis import (
    compare, deserialize, extract, index_groups, prepare_for_scoring, report, serialize, vector_keys,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS channels (
    source INTEGER PRIMARY KEY,
    destination INTEGER NOT NULL,
    discussion INTEGER NOT NULL,
    cursor INTEGER NOT NULL DEFAULT 0,
    snapshot INTEGER,
    history_done INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS posts (
    source INTEGER NOT NULL REFERENCES channels(source),
    id INTEGER NOT NULL,
    published TEXT NOT NULL,
    original TEXT NOT NULL,
    normalized TEXT NOT NULL,
    features BLOB NOT NULL,
    wording INTEGER NOT NULL,
    topic INTEGER NOT NULL,
    meaning INTEGER NOT NULL,
    style INTEGER NOT NULL,
    forward_random INTEGER NOT NULL UNIQUE,
    report_random INTEGER NOT NULL UNIQUE,
    forward_id INTEGER,
    discussion_root INTEGER,
    report_id INTEGER,
    skipped TEXT,
    PRIMARY KEY (source, id)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS postings (
    source INTEGER NOT NULL,
    metric INTEGER NOT NULL,
    key INTEGER NOT NULL,
    post INTEGER NOT NULL,
    PRIMARY KEY (source, metric, key, post)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS pending ON posts (source, id)
    WHERE report_id IS NULL AND skipped IS NULL;
CREATE TABLE IF NOT EXISTS copied_messages (
    source INTEGER NOT NULL,
    destination INTEGER NOT NULL,
    post INTEGER NOT NULL,
    published_id INTEGER NOT NULL,
    PRIMARY KEY (source, destination, post)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS route_settings (
    source INTEGER PRIMARY KEY REFERENCES channels(source),
    peers TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS manual_jobs (
    sender INTEGER NOT NULL,
    request_id INTEGER NOT NULL,
    source INTEGER NOT NULL,
    post INTEGER NOT NULL,
    result TEXT NOT NULL,
    forward_random INTEGER NOT NULL,
    report_random INTEGER NOT NULL,
    forward_id INTEGER,
    discussion_root INTEGER,
    report_id INTEGER,
    skipped TEXT,
    PRIMARY KEY(sender, request_id)
);
"""

STATUS_SQL = """
SELECT c.source, c.destination, c.cursor, c.snapshot, c.history_done,
       c.scan_count, c.index_limit, c.reply_until, c.auto_publish,
       coalesce(r.active, 0) AS active, coalesce(r.delivery_enabled, 0) AS delivery_enabled,
       count(p.id) AS indexed,
       count(p.report_id) + (SELECT count(*) FROM manual_jobs m
                             WHERE m.source = c.source AND m.report_id IS NOT NULL) AS delivered,
       count(p.skipped) + (SELECT count(*) FROM manual_jobs m
                           WHERE m.source = c.source AND m.skipped IS NOT NULL) AS skipped,
       sum(CASE WHEN p.publish_requested = 1 AND p.report_id IS NULL AND p.skipped IS NULL THEN 1 ELSE 0 END)
       + (SELECT count(*) FROM manual_jobs m WHERE m.source = c.source
          AND m.report_id IS NULL AND m.skipped IS NULL) AS pending
FROM channels c LEFT JOIN posts p ON p.source = c.source
LEFT JOIN route_settings r ON r.source = c.source
GROUP BY c.source ORDER BY c.source
"""


class Archive:
    def __init__(self, path: Path, analyzer_id='classic-v1', encoder=None):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = FULL;
            PRAGMA foreign_keys = ON;
            PRAGMA cache_size = -24000;
            PRAGMA busy_timeout = 5000;
        """)
        self.db.executescript(SCHEMA)
        self.encoder = encoder
        with self.db as db:
            db.execute('INSERT OR IGNORE INTO metadata VALUES (?, ?)', ('analyzer', analyzer_id))
            columns = {row['name'] for row in db.execute('PRAGMA table_info(posts)')}
            if 'embedding_ready' not in columns:
                db.execute('ALTER TABLE posts ADD COLUMN embedding_ready INTEGER NOT NULL DEFAULT 0')
            if 'copy_required' not in columns:
                db.execute('ALTER TABLE posts ADD COLUMN copy_required INTEGER NOT NULL DEFAULT 0')
            if 'publish_requested' not in columns:
                db.execute('ALTER TABLE posts ADD COLUMN publish_requested INTEGER NOT NULL DEFAULT 1')
            channel_columns = {row['name'] for row in db.execute('PRAGMA table_info(channels)')}
            if 'scan_count' not in channel_columns:
                db.execute('ALTER TABLE channels ADD COLUMN scan_count INTEGER NOT NULL DEFAULT 0')
            if 'index_limit' not in channel_columns:
                db.execute('ALTER TABLE channels ADD COLUMN index_limit INTEGER')
            if 'reply_until' not in channel_columns:
                db.execute('ALTER TABLE channels ADD COLUMN reply_until REAL NOT NULL DEFAULT 0')
            if 'auto_publish' not in channel_columns:
                db.execute('ALTER TABLE channels ADD COLUMN auto_publish INTEGER NOT NULL DEFAULT 1')
            route_columns = {row['name'] for row in db.execute('PRAGMA table_info(route_settings)')}
            if 'delivery_enabled' not in route_columns:
                db.execute('ALTER TABLE route_settings ADD COLUMN delivery_enabled INTEGER NOT NULL DEFAULT 1')
            db.execute('CREATE INDEX IF NOT EXISTS embeddings_pending ON posts (source, id) WHERE embedding_ready = 0')
        if encoder is not None:
            self.set_encoder(encoder)

    def close(self):
        self.db.close()

    def pin(self, key, value):
        value = str(value)
        with self.db as db:
            db.execute('INSERT OR IGNORE INTO metadata VALUES (?, ?)', (key, value))
            old = db.execute('SELECT value FROM metadata WHERE key = ?', (key,)).fetchone()
            if old[0] != value:
                raise ValueError(f'Настройка {key} не совпадает с сохранённой в БД')

    def setting(self, key, default=None):
        row = self.db.execute('SELECT value FROM metadata WHERE key = ?', (key,)).fetchone()
        return row[0] if row else default

    def set_setting(self, key, value):
        with self.db as db:
            db.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', (key, str(value)))

    def set_encoder(self, encoder):
        if encoder is not None:
            previous = self.setting('analyzer', '')
            if previous.startswith('minilm-') and previous != encoder.identity:
                raise ValueError('Файлы MiniLM отличаются от использованных в этой базе')
            self.pin('semantic_encoder', encoder.identity)
        self.encoder = encoder

    def missing_embedding(self, source):
        return self.db.execute(
            'SELECT id, original, features FROM posts WHERE source = ? AND embedding_ready = 0 ORDER BY id LIMIT 1',
            (source,),
        ).fetchone()

    @staticmethod
    def unpack_features(blob):
        return json.loads(zlib.decompress(blob))

    def store_embedding(self, source, post, features):
        keys = {key for unit in features['units'] for key in vector_keys(unit['vector'])}
        with self.db as db:
            db.execute('UPDATE posts SET features = ?, embedding_ready = 1 WHERE source = ? AND id = ?',
                       (serialize(features), source, post))
            db.executemany('INSERT OR IGNORE INTO postings VALUES (?, 5, ?, ?)',
                           ((source, key, post) for key in keys))

    def status_rows(self):
        return self.db.execute(STATUS_SQL)

    def route_records(self):
        for row in self.db.execute('SELECT source, peers, active FROM route_settings'):
            yield row['source'], json.loads(row['peers']), row['active']

    def save_route(self, source, destination, discussion, peers, *, skip_pending=False, active=1):
        old = self.state(source)
        changed = old is not None and (old['destination'], old['discussion']) != (destination, discussion)
        with self.db as db:
            if changed:
                if self.queue_size(source):
                    if not skip_pending:
                        raise ValueError('Есть посты в очереди прежнего назначения. Используй /drain или явный skip_pending.')
                    db.execute("""
                        UPDATE posts SET skipped = 'очередь пропущена при смене назначения'
                        WHERE source = ? AND publish_requested = 1 AND report_id IS NULL AND skipped IS NULL
                    """, (source,))
                    db.execute("""
                        UPDATE manual_jobs SET skipped = 'очередь пропущена при смене назначения'
                        WHERE source = ? AND report_id IS NULL AND skipped IS NULL
                    """, (source,))
                db.execute('UPDATE channels SET destination = ?, discussion = ? WHERE source = ?',
                           (destination, discussion, source))
            elif old is None:
                db.execute('INSERT INTO channels (source, destination, discussion, auto_publish) '
                           'VALUES (?, ?, ?, 0)', (source, destination, discussion))
            exists = db.execute('SELECT 1 FROM route_settings WHERE source = ?', (source,)).fetchone()
            db.execute('INSERT INTO route_settings (source, peers, active, delivery_enabled) VALUES (?, ?, ?, 1) '
                       'ON CONFLICT(source) DO UPDATE SET peers = excluded.peers, active = excluded.active',
                       (source, json.dumps(peers), active))
            if not exists and not changed:
                db.execute("""
                    UPDATE posts SET skipped = NULL, copy_required = 1
                    WHERE source = ? AND skipped = 'Telegram запретил пересылку' AND forward_id IS NULL
                """, (source,))

    def set_route_active(self, source, active):
        with self.db as db:
            if not db.execute('UPDATE route_settings SET active = ? WHERE source = ?', (active, source)).rowcount:
                raise ValueError('Маршрут не настроен')

    def delivery_enabled(self, source):
        return self.db.execute('SELECT delivery_enabled FROM route_settings WHERE source = ?',
                               (source,)).fetchone()[0]

    def set_delivery_enabled(self, source, enabled):
        with self.db as db:
            db.execute('UPDATE route_settings SET delivery_enabled = ? WHERE source = ?', (enabled, source))

    def set_auto_publish(self, source, enabled):
        with self.db as db:
            db.execute('UPDATE channels SET auto_publish = ? WHERE source = ?', (enabled, source))

    def set_index_limit(self, source, limit):
        with self.db as db:
            db.execute('UPDATE channels SET index_limit = ? WHERE source = ?', (limit, source))

    def start_from_now(self, source, head):
        with self.db as db:
            db.execute('UPDATE channels SET cursor = MAX(cursor, ?), snapshot = NULL, '
                       'history_done = 1, index_limit = NULL WHERE source = ?', (head, source))

    def set_reply_until(self, source, until):
        with self.db as db:
            db.execute('UPDATE channels SET reply_until = ? WHERE source = ?', (until, source))

    def reply_until(self, source):
        return self.state(source)['reply_until']

    def remember_error(self, source, message):
        self.set_setting(f'error:{source}', message[:1000])

    def error(self, source):
        return self.setting(f'error:{source}')

    def flood_until(self, identity):
        row = self.db.execute('SELECT value FROM metadata WHERE key = ?',
                              (f'flood:{identity}',)).fetchone()
        return float(row[0]) if row else 0

    def defer_flood(self, identity, until):
        with self.db as db:
            db.execute('''
                INSERT INTO metadata VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = MAX(CAST(value AS REAL), CAST(excluded.value AS REAL))
            ''', (f'flood:{identity}', str(until)))

    def register(self, source, destination, discussion):
        with self.db as db:
            db.execute('INSERT OR IGNORE INTO channels (source, destination, discussion) VALUES (?, ?, ?)',
                       (source, destination, discussion))
            route = self.state(source)
            if (route['destination'], route['discussion']) != (destination, discussion):
                raise ValueError(f'Маршрут для {source} отличается от сохранённого в БД')

    def state(self, source):
        return self.db.execute('SELECT * FROM channels WHERE source = ?', (source,)).fetchone()

    def snapshot(self, source, head):
        with self.db as db:
            db.execute('UPDATE channels SET snapshot = ? WHERE source = ? AND snapshot IS NULL',
                       (head, source))

    def finish_snapshot(self, source, head):
        with self.db as db:
            db.execute('UPDATE channels SET cursor = MAX(cursor, ?), snapshot = NULL, history_done = 1 '
                       'WHERE source = ? AND snapshot = ?', (head, source, head))

    def index(self, source, post, published, original):
        with self.db as db:
            if self.state(source)['cursor'] >= post:
                return
            if original and original.strip():
                features = extract(original, self.encoder)
                packed = serialize(features)
                keys = {(metric, key) for metric, group in index_groups(features) for key in group}
                probes = list(index_groups(features, query=True))
                current = prepare_for_scoring(features)
                percentages = self.score(source, post, current, probes)
                db.execute("""
                    INSERT INTO posts (source, id, published, original, normalized, features,
                                       wording, topic, meaning, style, forward_random, report_random,
                                       embedding_ready, publish_requested)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (source, post, published, original, features['normalized'], packed,
                      *percentages, self.random_id(), self.random_id(), self.encoder is not None,
                      self.state(source)['auto_publish']))
                db.executemany('INSERT INTO postings VALUES (?, ?, ?, ?)',
                               ((source, metric, key, post) for metric, key in keys))
            db.execute('UPDATE channels SET cursor = ?, scan_count = scan_count + 1 WHERE source = ?',
                       (post, source))

    @staticmethod
    def random_id():
        return secrets.randbits(63) or 1

    def score(self, source, post, current, probes):
        similarities = [0.0] * 4
        for candidate in self.candidates(source, post, probes):
            with closing(self.db.execute('SELECT features FROM posts WHERE source = ? AND id = ?',
                                         (source, candidate))) as cursor:
                previous = deserialize(cursor.fetchone()[0])
            if 'vectors' in current and 'vectors' not in previous and self.encoder:
                original = self.db.execute('SELECT original FROM posts WHERE source = ? AND id = ?',
                                           (source, candidate)).fetchone()[0]
                previous = prepare_for_scoring(extract(original, self.encoder))
            compare(current, previous, similarities)
            if min(similarities) >= 1:
                break
        return tuple(round(100 * (1 - min(1.0, value))) for value in similarities)

    def candidates(self, source, post, probes):
        by_metric = {}
        for metric, keys in probes:
            votes = {}
            for key in keys:
                hits = self.db.execute("""
                    SELECT post FROM postings WHERE source = ? AND metric = ? AND key = ? AND post < ?
                    ORDER BY post DESC LIMIT 25
                """, (source, metric, key, post)).fetchall()
                if not hits:
                    continue
                weight = 1 / len(hits) ** 0.5
                for (match,) in hits:
                    votes[match] = votes.get(match, 0) + weight
            best = sorted(votes, key=lambda match: (votes[match], match), reverse=True)[:3]
            totals = by_metric.setdefault(metric, {})
            for match in best:
                totals[match] = max(totals.get(match, 0), votes[match])
        seen = set()
        for metric in (4, 0, 1, 2, 5, 3):
            votes = by_metric.get(metric, {})
            for match in sorted(votes, key=lambda m: (votes[m], m), reverse=True)[:12]:
                if match not in seen:
                    seen.add(match)
                    yield match
        recent = self.db.execute('SELECT id FROM posts WHERE source = ? AND id < ? ORDER BY id DESC LIMIT 8',
                                 (source, post))
        for (match,) in recent:
            if match not in seen:
                yield match

    def copied_id(self, source, destination, post):
        row = self.db.execute(
            'SELECT published_id FROM copied_messages WHERE source = ? AND destination = ? AND post = ?',
            (source, destination, post),
        ).fetchone()
        return row['published_id'] if row else None

    def store_copies(self, source, destination, copies):
        with self.db as db:
            db.executemany('INSERT OR IGNORE INTO copied_messages VALUES (?, ?, ?, ?)',
                           ((source, destination, post, published) for post, published in copies.items()))

    def next_job(self, source):
        return self.db.execute("""
            SELECT source, id, wording, topic, meaning, style, forward_random, report_random,
                   forward_id, discussion_root, copy_required
            FROM posts WHERE source = ? AND publish_requested = 1 AND report_id IS NULL AND skipped IS NULL
            ORDER BY id LIMIT 1
        """, (source,)).fetchone()

    def next_forward_job(self, source):
        return self.db.execute("""
            SELECT source, id, wording, topic, meaning, style, forward_random, report_random,
                   forward_id, discussion_root, copy_required
            FROM posts WHERE source = ? AND publish_requested = 1 AND forward_id IS NULL
              AND report_id IS NULL AND skipped IS NULL
            ORDER BY id LIMIT 1
        """, (source,)).fetchone()

    def queue_size(self, source):
        automatic = self.db.execute('SELECT count(*) FROM posts WHERE source = ? AND publish_requested = 1 AND report_id IS NULL '
                               'AND skipped IS NULL', (source,)).fetchone()[0]
        manual = self.db.execute('SELECT count(*) FROM manual_jobs WHERE source = ? AND report_id IS NULL '
                                 'AND skipped IS NULL', (source,)).fetchone()[0]
        return automatic + manual

    def add_manual(self, sender, request_id, source, post, result):
        with self.db as db:
            db.execute('INSERT OR IGNORE INTO manual_jobs '
                       '(sender, request_id, source, post, result, forward_random, report_random) '
                       'VALUES (?, ?, ?, ?, ?, ?, ?)',
                       (sender, request_id, source, post, result, self.random_id(), self.random_id()))

    def next_manual(self, source, *, forward_only=False):
        extra = 'AND forward_id IS NULL' if forward_only else ''
        return self.db.execute(f'SELECT * FROM manual_jobs WHERE source = ? AND report_id IS NULL '
                               f'AND skipped IS NULL {extra} ORDER BY request_id LIMIT 1', (source,)).fetchone()

    def update_manual(self, sender, request_id, *, forward=None, root=None, result=None, skipped=None):
        with self.db as db:
            db.execute('UPDATE manual_jobs SET forward_id = COALESCE(?, forward_id), '
                       'discussion_root = COALESCE(?, discussion_root), '
                       'report_id = COALESCE(?, report_id), skipped = COALESCE(?, skipped) '
                       'WHERE sender = ? AND request_id = ?',
                       (forward, root, result, skipped, sender, request_id))

    def summary(self, source, limit=None):
        sql = ('SELECT count(*), avg(wording), avg(topic), avg(meaning), avg(style) FROM '
               '(SELECT wording, topic, meaning, style FROM posts WHERE source = ? ORDER BY id'
               + (' LIMIT ?' if limit else '') + ')')
        row = self.db.execute(sql, (source, limit) if limit else (source,)).fetchone()
        if not row[0]:
            return 0, (0, 0, 0, 0)
        return row[0], tuple(round(value) for value in row[1:])

    def indexed_count(self, source):
        return self.db.execute('SELECT count(*) FROM posts WHERE source = ?', (source,)).fetchone()[0]

    def update_delivery(self, source, post, *, forward=None, root=None, result=None, skipped=None, copy_required=None):
        with self.db as db:
            db.execute("""
                UPDATE posts SET forward_id = COALESCE(?, forward_id),
                                 discussion_root = COALESCE(?, discussion_root),
                                 report_id = COALESCE(?, report_id),
                                 skipped = COALESCE(?, skipped),
                                 copy_required = COALESCE(?, copy_required)
                WHERE source = ? AND id = ?
            """, (forward, root, result, skipped, copy_required, source, post))

    @staticmethod
    def report_for(job):
        return report([job[metric] for metric in ('wording', 'topic', 'meaning', 'style')])

def read_status(path):
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(STATUS_SQL)]
