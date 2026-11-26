"""Word, phrase and semantic repetition within one channel."""

import json
import hashlib
import re
import unicodedata
import zlib
from collections import Counter
from functools import lru_cache

import snowballstemmer

WORD = re.compile(r'[^\W_]+', re.UNICODE)
STOP = set('а и в во на по из к ко за с со о об от до для у я мы вы он она они это как что '
           'чтобы или но если уже же то еще есть меня мне мой наш его ее их те тот там тут '
           'the a an is are was were be to of in on for and or that this with as at by it'.split())
RU = snowballstemmer.stemmer('russian')
EN = snowballstemmer.stemmer('english')


@lru_cache(maxsize=16384)
def stem(word):
    if re.fullmatch('[а-я]+', word):
        return RU.stemWord(word)
    if re.fullmatch('[a-z]+', word):
        return EN.stemWord(word)
    return word


def words(text):
    normalized = unicodedata.normalize('NFKC', text).casefold().replace('ё', 'е')
    return WORD.findall(normalized)


def stable_hash(value):
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), 'big') & ((1 << 63) - 1)


def content(text):
    return [stem(word) for word in words(text) if word not in STOP and not word.isdecimal()]


def phrases(items):
    return [tuple(items[pos:pos + size]) for size in (2, 3)
            for pos in range(len(items) - size + 1)]


def fragments(text):
    """Short spans keep long posts from being cut at 128 model tokens."""
    items = words(text)
    if len(items) <= 36:
        return [' '.join(items)] if items else []
    return [' '.join(items[pos:pos + 36]) for pos in range(0, len(items), 28)]


def extract(text, encoder=None):
    return {'vectors': list(encoder.encode(fragments(text)))} if encoder else {}


def serialize(features):
    return zlib.compress(json.dumps(features, separators=(',', ':')).encode(), 3)


def deserialize(blob):
    return json.loads(zlib.decompress(blob))


def novelty(items, counts, owned, total, common):
    if not items or not total:
        return 100
    return round(100 * sum(counts[item] - (item in owned) < common for item in items) / len(items))


def word_novelty(items, counts, owned, total):
    if not items or not total:
        return 100
    common = max(2, (total + 49) // 50)
    rarity = {item: min(1, max(0, 1 - (counts[item] - (item in owned) - 1) / (common - 1)))
              for item in items}
    familiar = sorted(rarity.values())[:4]
    return round(100 * (0.7 * sum(familiar) / len(familiar)
                        + 0.3 * sum(rarity[item] for item in items) / len(items)))


def corpus_counts(rows):
    terms, combinations = Counter(), Counter()
    per_post = {}
    for row in rows:
        tokens = content(row['original'])
        own_terms, own_phrases = set(tokens), set(phrases(tokens))
        terms.update(own_terms)
        combinations.update(own_phrases)
        per_post[row['id']] = (own_terms, own_phrases)
    return terms, combinations, per_post


def semantic(query_vectors, rows, excluded=None):
    """Share of query fragments without a close fragment anywhere in the channel."""
    import numpy as np

    query = np.asarray(query_vectors, dtype=np.float32)
    if not query.size:
        return None
    query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1)
    vectors = []
    for row in rows:
        if row['id'] == excluded:
            continue
        stored = deserialize(row['features']).get('vectors', [])
        vectors.extend(stored)
    if not vectors:
        return None
    reference = np.asarray(vectors, dtype=np.float32)
    reference /= np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1)
    closest = query @ reference.T
    best = closest.max(axis=1)
    return round(100 * np.clip((0.82 - best) / 0.22, 0, 1).mean())
