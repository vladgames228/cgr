import secrets
import sqlite3
from contextlib import closing
from pathlib import Path

from .analysis import (
    compare, deserialize, extract, index_groups, prepare_for_scoring, report, serialize,
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
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

STATUS_SQL = """
SELECT c.source, c.destination, c.cursor, c.snapshot, c.history_done,
       count(p.id) AS indexed,
       count(p.report_id) AS delivered,
       count(p.skipped) AS skipped,
       count(p.id) - count(p.report_id) - count(p.skipped) AS pending
FROM channels c LEFT JOIN posts p ON p.source = c.source
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
        self.pin('analyzer', analyzer_id)

    def close(self):
        self.db.close()

    def pin(self, key, value):
        value = str(value)
        with self.db as db:
            db.execute('INSERT OR IGNORE INTO metadata VALUES (?, ?)', (key, value))
            old = db.execute('SELECT value FROM metadata WHERE key = ?', (key,)).fetchone()
            if old[0] != value:
                raise ValueError(f'Настройка {key} не совпадает с сохранённой в БД')

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
                                       wording, topic, meaning, style, forward_random, report_random)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (source, post, published, original, features['normalized'], packed,
                      *percentages, self.random_id(), self.random_id()))
                db.executemany('INSERT INTO postings VALUES (?, ?, ?, ?)',
                               ((source, metric, key, post) for metric, key in keys))
            db.execute('UPDATE channels SET cursor = ? WHERE source = ?', (post, source))

    @staticmethod
    def random_id():
        return secrets.randbits(63) or 1

    def score(self, source, post, current, probes):
        similarities = [0.0] * 4
        for candidate in self.candidates(source, post, probes):
            with closing(self.db.execute('SELECT features FROM posts WHERE source = ? AND id = ?',
                                         (source, candidate))) as cursor:
                previous = deserialize(cursor.fetchone()[0])
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

    def next_job(self, source):
        return self.db.execute("""
            SELECT source, id, wording, topic, meaning, style, forward_random, report_random,
                   forward_id, discussion_root
            FROM posts WHERE source = ? AND report_id IS NULL AND skipped IS NULL
            ORDER BY id LIMIT 1
        """, (source,)).fetchone()

    def update_delivery(self, source, post, *, forward=None, root=None, result=None, skipped=None):
        with self.db as db:
            db.execute("""
                UPDATE posts SET forward_id = COALESCE(?, forward_id),
                                 discussion_root = COALESCE(?, discussion_root),
                                 report_id = COALESCE(?, report_id),
                                 skipped = COALESCE(?, skipped)
                WHERE source = ? AND id = ?
            """, (forward, root, result, skipped, source, post))

    @staticmethod
    def report_for(job):
        return report([job[metric] for metric in ('wording', 'topic', 'meaning', 'style')])

def read_status(path):
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(STATUS_SQL)]
