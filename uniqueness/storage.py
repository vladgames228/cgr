import json
import hashlib
from itertools import islice
import os
import secrets
import sqlite3
from contextlib import closing
from pathlib import Path

from .analysis import (content, corpus_counts, deserialize, extract, novelty, phrases, semantic,
                       serialize, style_novelty, style_profile, word_novelty, words)

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
    draft_id INTEGER,
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
CREATE TABLE IF NOT EXISTS groups (
    id INTEGER PRIMARY KEY,
    peer TEXT NOT NULL,
    name TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    cursor INTEGER NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS group_results (
    group_id INTEGER NOT NULL,
    post INTEGER NOT NULL,
    result TEXT NOT NULL,
    PRIMARY KEY (group_id, post)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS fingerprints (
    source INTEGER NOT NULL,
    digest BLOB NOT NULL,
    PRIMARY KEY (source, digest)
) WITHOUT ROWID;
"""


class Archive:
    def __init__(self, path: Path, analyzer_id='content-style-v1', encoder=None):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;')
        existing = self.db.execute("SELECT 1 FROM sqlite_master WHERE name='channels'").fetchone()
        if existing and 'peers' not in {row['name'] for row in self.db.execute('PRAGMA table_info(channels)')}:
            self.db.close()
            raise ValueError('Старая схема базы. Останови контейнер и очисти archive.sqlite3 перед запуском cgr.')
        self.db.executescript(SCHEMA)
        if 'draft_id' not in {row['name'] for row in self.db.execute('PRAGMA table_info(outbox)')}:
            self.db.execute('ALTER TABLE outbox ADD COLUMN draft_id INTEGER')
        self.encoder = encoder
        self._corpus_cache = {}
        self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', ('analyzer', analyzer_id))
        self.db.execute('INSERT OR IGNORE INTO metadata VALUES (?, ?)', ('min_words', '3'))
        empty_posts = [(row['source'], row['id']) for row in
                       self.db.execute('SELECT source, id, original FROM posts')
                       if not words(row['original'])]
        self.db.executemany('DELETE FROM posts WHERE source = ? AND id = ?', empty_posts)
        if not self.db.execute('SELECT 1 FROM fingerprints LIMIT 1').fetchone():
            self.db.executemany('INSERT OR IGNORE INTO fingerprints VALUES (?, ?)',
                                ((row['source'], self._fingerprint(row['original'])) for row in
                                 self.db.execute('SELECT source, original FROM posts WHERE source IN (SELECT id FROM groups)')))
        self.db.commit()

    @staticmethod
    def _fingerprint(text):
        return hashlib.blake2b(' '.join(words(text)).encode(), digest_size=16).digest()

    def apply_min_words(self, minimum):
        self.set_setting('min_words', minimum)

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

    def groups(self):
        return list(self.db.execute('SELECT * FROM groups WHERE active >= 0 ORDER BY name'))

    def group(self, group_id):
        return self.db.execute('SELECT * FROM groups WHERE id = ?', (group_id,)).fetchone()

    def save_group(self, group_id, peer, name, user_id, cursor):
        with self.db as db:
            db.execute('INSERT INTO groups (id, peer, name, user_id, cursor) VALUES (?, ?, ?, ?, ?) '
                       'ON CONFLICT(id) DO UPDATE SET peer=excluded.peer, name=excluded.name, '
                       'user_id=excluded.user_id, active=1',
                       (group_id, json.dumps(peer), name, user_id, cursor))

    def remove_group(self, group_id):
        with self.db as db:
            db.execute('UPDATE groups SET active = -1 WHERE id = ?', (group_id,))

    def group_cursor(self, group_id, post):
        with self.db as db:
            db.execute('UPDATE groups SET cursor = MAX(cursor, ?) WHERE id = ?', (post, group_id))

    def group_result(self, group_id, post, result):
        with self.db as db:
            db.execute('INSERT OR IGNORE INTO group_results VALUES (?, ?, ?)', (group_id, post, result))

    def recent_group_results(self, group_id, limit=5):
        return list(self.db.execute('SELECT post, result FROM group_results WHERE group_id = ? '
                                    'ORDER BY post DESC LIMIT ?', (group_id, limit)))

    def state(self, source):
        return self.db.execute('SELECT * FROM channels WHERE source = ?', (source,)).fetchone()

    def remove_route(self, source):
        with self.db as db:
            db.execute('DELETE FROM outbox WHERE source = ? AND report_id IS NULL', (source,))
            db.execute('DELETE FROM channels WHERE source = ?', (source,))

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
            db.execute('DELETE FROM fingerprints WHERE source = ?', (source,))
            db.execute('UPDATE channels SET cursor = 0, snapshot = NULL, history_done = 0, '
                       'scan_count = 0, active = 1 WHERE source = ?', (source,))
        self._corpus_cache.pop(source, None)
        self._corpus_cache.pop(None, None)

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
            db.execute('UPDATE groups SET active = 0')

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
            if original and len(words(original)) >= int(self.setting('min_words', 3)):
                self._store_post(source, post, original)
            db.execute('UPDATE channels SET cursor = ?, scan_count = scan_count + 1 WHERE source = ?',
                       (post, source))

    def add_reference(self, source, post, original, *, encode=True):
        if self.state(source) is None and self.group(source) is None:
            raise ValueError('Источник не настроен')
        if not original or not words(original):
            raise ValueError('У поста нет слов для индексации')
        with self.db:
            if post is None:
                row = self.db.execute('SELECT MIN(id) FROM posts WHERE source = ? AND id < 0',
                                      (source,)).fetchone()
                post = min(0, row[0] or 0) - 1
            return self._store_post(source, post, original, encode=encode)

    def _store_post(self, source, post, original, *, encode=True):
        if not words(original):
            return False
        if self.db.execute('SELECT 1 FROM posts WHERE source = ? AND id = ?', (source, post)).fetchone():
            return False
        digest = self._fingerprint(original)
        if self.group(source) and self.db.execute(
                'SELECT 1 FROM fingerprints WHERE source = ? AND digest = ?',
                (source, digest)).fetchone():
            return False
        packed = serialize(extract(original, self.encoder if encode else None))
        self.db.execute('INSERT INTO posts VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                        (source, post, original, packed, 0, 0, 0, 0))
        if self.group(source):
            self.db.execute('INSERT OR IGNORE INTO fingerprints VALUES (?, ?)', (source, digest))
        self._corpus_cache.pop(source if self.group(source) else None, None)
        return True

    def fill_vectors(self, source, batch_size=4):
        if self.encoder is None:
            return False
        missing = []
        query = ('SELECT source, id, original, features FROM posts WHERE source = ?' if self.group(source)
                 else 'SELECT source, id, original, features FROM posts WHERE source NOT IN (SELECT id FROM groups)')
        parameters = (source,) if self.group(source) else ()
        for row in self.db.execute(query, parameters):
            if words(row['original']) and not deserialize(row['features']).get('vectors'):
                missing.append(row)
                if len(missing) == batch_size:
                    break
        if not missing:
            return False
        with self.db as db:
            for row in missing:
                db.execute('UPDATE posts SET features = ? WHERE source = ? AND id = ?',
                           (serialize(extract(row['original'], self.encoder)), row['source'], row['id']))
        self._corpus_cache.pop(source if self.group(source) else None, None)
        return True

    def _corpus(self, source):
        key = source if self.group(source) else None
        if key not in self._corpus_cache:
            if key is None:
                all_rows = self.db.execute('SELECT source, id, original, features FROM posts '
                                           'WHERE source NOT IN (SELECT id FROM groups) ORDER BY source, id')
                seen = set()
                rows = []
                exact = {}
                for row in all_rows:
                    digest = self._fingerprint(row['original'])
                    exact.setdefault(digest, []).append((row['source'], row['id']))
                    if digest not in seen:
                        seen.add(digest)
                        rows.append(row)
            else:
                rows = list(self.db.execute('SELECT source, id, original, features FROM posts WHERE source = ?',
                                            (source,)))
                exact = {}
                for row in rows:
                    exact.setdefault(self._fingerprint(row['original']), []).append((row['source'], row['id']))
            self._corpus_cache[key] = (rows, *corpus_counts(rows), exact, style_profile(rows))
        return self._corpus_cache[key]

    def analyze(self, source, text, exclude=None):
        rows, term_counts, phrase_counts, per_post, exact, profile = self._corpus(source)
        excluded = (source, exclude) if exclude is not None else None
        others = len(rows) - (excluded in per_post)
        terms = content(text)
        combinations = phrases(terms)
        owned_terms, owned_phrases = per_post.get(excluded, (set(), set()))
        for post_key in exact.get(self._fingerprint(text), ()):
            if post_key != excluded:
                return {'overall': 0, 'words': 0, 'phrases': 0,
                        'theme': 0 if self.encoder else None, 'content': 0, 'style': 0,
                        'model_pending': False, 'similar_post': post_key, 'similarity': 100}
        if not others:
            return {'overall': 100, 'words': 100, 'phrases': 100, 'theme': None,
                    'content': 100, 'style': None,
                    'model_pending': False, 'similar_post': None, 'similarity': 0}
        vocabulary = word_novelty(terms, term_counts, owned_terms, others)
        expressions = novelty(combinations, phrase_counts, owned_phrases, others)
        theme, pending = None, False
        if self.encoder:
            pending = any(words(row['original']) and not deserialize(row['features']).get('vectors')
                          for row in rows if (row['source'], row['id']) != excluded)
            if not pending:
                theme = semantic(extract(text, self.encoder)['vectors'], rows, excluded)
        lexical = 0.65 * vocabulary + 0.35 * expressions
        # A paraphrase should remain a content repeat even when none of its terms match.
        content_score = round(lexical if theme is None else 0.8 * theme + 0.2 * min(theme, lexical))
        if excluded is not None and profile is not None:
            profile = style_profile([row for row in rows if (row['source'], row['id']) != excluded])
        style_score = style_novelty(text, profile)
        overall = content_score if style_score is None else round((content_score + style_score) / 2)
        if (terms and len(terms) <= 4
                and all(term_counts[term] - (term in owned_terms) >= 1 for term in terms)
                and (not combinations or any(phrase_counts[part] - (part in owned_phrases) >= 1
                                             for part in combinations))):
            content_score = 0
            overall = 0 if style_score is None else round(style_score / 2)
        own_words, own_phrases = set(terms), set(combinations)
        similar_post, similarity = None, 0
        if own_words:
            for post, (other_words, other_phrases) in per_post.items():
                if post == excluded or not other_words:
                    continue
                word_match = len(own_words & other_words) / len(own_words | other_words)
                phrase_match = (len(own_phrases & other_phrases) / len(own_phrases | other_phrases)
                                if own_phrases and other_phrases else 0)
                match = round(100 * ((word_match + phrase_match) / 2 if own_phrases else word_match))
                if match > similarity:
                    similar_post, similarity = post, match
        return {'overall': overall, 'words': vocabulary, 'phrases': expressions,
                'theme': theme, 'content': content_score, 'style': style_score,
                'model_pending': pending,
                'similar_post': similar_post, 'similarity': similarity}

    def statistics(self, source):
        rows, terms, expressions, _, _, _ = self._corpus(source)
        useful = ((term, count) for term, count in terms.most_common() if len(term) >= 3)
        useful_phrases = ((phrase, count) for phrase, count in expressions.most_common()
                          if count > 1 and all(len(term) >= 3 for term in phrase))
        return len(rows), list(islice(useful, 8)), list(islice(useful_phrases, 5))

    def post_text(self, source, post):
        row = self.db.execute('SELECT original FROM posts WHERE source = ? AND id = ?',
                              (source, post)).fetchone()
        return row['original'] if row else None

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

    def update_job(self, source, post, *, forward=None, result=None, skipped=None, draft=None):
        with self.db as db:
            db.execute('UPDATE outbox SET forward_id = COALESCE(?, forward_id), '
                       'report_id = COALESCE(?, report_id), skipped = COALESCE(?, skipped), '
                       'draft_id = COALESCE(?, draft_id) WHERE source = ? AND post = ?',
                       (forward, result, skipped, draft, source, post))

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
