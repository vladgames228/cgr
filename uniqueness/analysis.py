import hashlib
import json
import math
import re
import unicodedata
import zlib
from collections import Counter
from functools import lru_cache

import snowballstemmer

LABELS = ('формулировок', 'тематики', 'смысла', 'стиля')
WORD = re.compile(r'[^\W_]+', re.UNICODE)
SENTENCE = re.compile(r'(?<=[.!?…])\s+|\n+')
FIRST_PERSON = set('я мы мне меня нам мой наш i me my we us our'.split())
EXPRESSIVE = set('очень ужасно отлично круто увы жаль ну прямо реально very really wow awesome sadly'.split())
NEGATION = set('не нет нельзя никогда без ни no not never without cannot'.split())
STOP = NEGATION | set('а и в во на по из к ко за с со о об от до для у я мы вы он она они это как что чтобы или но если уже же то еще есть the a an is are was were be been to of in on for and or that this with as at by it he she we they i you from'.split())
RU = snowballstemmer.stemmer('russian')
EN = snowballstemmer.stemmer('english')
CYRILLIC = re.compile(r'^[а-я]+$')
LATIN = re.compile(r'^[a-z]+$')


@lru_cache(maxsize=8192)
def stem(word):
    if CYRILLIC.fullmatch(word):
        return RU.stemWord(word)
    if LATIN.fullmatch(word):
        return EN.stemWord(word)
    return word


SYNONYMS = {}
for group in (
    'купить приобрести покупка приобретение buy purchase',
    'продать продавать продажа sell',
    'увеличить рост вырасти повысить подорожать повышение increase rise grow',
    'уменьшить снижение снизить упасть подешеветь decrease fall decline',
    'цена стоимость расценка price cost',
    'автомобиль машина авто car automobile',
    'работник сотрудник персонал employee worker',
    'запустить начать стартовать старт launch start begin',
    'закрыть завершить закончить прекратить close finish end',
    'сказать сообщить заявить объявить announce report',
    'разрешить позволить allow permit',
    'запретить запрет ban prohibit',
    'быстрый скорый стремительный rapid fast quick',
    'медленный неспешный slow',
    'компания фирма предприятие организация company firm business',
    'помощь поддержка помочь help support assist',
    'жильё квартира апартаменты home apartment',
    'врач доктор медик doctor physician',
    'заболевание болезнь недуг illness disease',
    'трудный сложный непростой difficult hard',
    'планировать намереваться собираться plan intend',
    'получить обрести receive obtain',
):
    words = group.replace('ё', 'е').split()
    for word in words:
        SYNONYMS[stem(word)] = words[0]


def tokens(text):
    return WORD.findall(text.casefold().replace('ё', 'е'))


def stable_hash(value):
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), 'big') & ((1 << 63) - 1)


def shingles(sequence, size):
    return [stable_hash(' '.join(sequence[i:i + size])) for i in range(len(sequence) - size + 1)]


def lexical(words, canonical):
    char = [stable_hash(canonical[i:i + 5]) for i in range(len(canonical) - 4)]
    return [sorted(set(shingles(words, size))) for size in (1, 2, 3)] + [sorted(set(char))]


def style(text, words, sentences):
    length = max(1, len(words))
    sentence_lengths = [len(tokens(sentence)) for sentence in sentences] or [0]
    return [
        min(3, math.log1p(sum(sentence_lengths) / len(sentence_lengths)) / 1.5),
        min(3, math.log1p(max(sentence_lengths)) / 1.7),
        min(3, sum(map(len, words)) / length / 3),
        2 * sum(word in STOP for word in words) / length,
        2 * sum(len(word) >= 8 for word in words) / length,
        2 * sum(len(word) <= 3 for word in words) / length,
        1.4 * len(set(words)) / length,
        min(2, 8 * text.count(',') / length),
        min(2, 8 * (text.count('!') + text.count('?')) / length),
        min(2, 6 * sum(text.count(char) for char in ':;—–()«»') / length),
        min(2, 10 * text.count('\n') / length),
        min(2, 5 * sum(char.isupper() for char in text) / max(1, len(text))),
        min(2, 5 * sum(unicodedata.category(char) == 'So' for char in text) / length),
        min(2, 5 * sum(word.isdecimal() for word in words) / length),
        min(2, 8 * sum(word in FIRST_PERSON for word in words) / length),
        min(2, 8 * sum(word in EXPRESSIVE for word in words) / length),
    ]


def parts(sentences):
    # Include every sentence, split long sentences, and join short neighbors.
    for sentence in sentences:
        spans = list(WORD.finditer(sentence))
        if len(spans) <= 36:
            if spans:
                yield sentence
            continue
        for offset in range(0, len(spans), 24):
            window = spans[offset:offset + 36]
            if offset and len(window) <= 12:
                break
            yield sentence[window[0].start():window[-1].end()]
    for left, right in zip(sentences, sentences[1:]):
        if 0 < len(tokens(left)) < 8 and 0 < len(tokens(right)) < 8:
            yield left + ' ' + right


def unit(text, words):
    canonical = ' '.join(text.casefold().replace('ё', 'е').split())
    content = [stem(word) for word in words if word not in STOP and not word.isdecimal()]
    concepts = [SYNONYMS.get(word, word) for word in content]
    return {
        'n': len(words), 'lex': lexical(words, canonical), 'topic': Counter(content),
        'concept': Counter(concepts), 'pairs': sorted(set(shingles(concepts, 2))),
        'neg': sum(word in NEGATION for word in words), 'numbers': sorted(set(
            word for word in words if word.isdecimal())),
        'raw': stable_hash(canonical),
    }


def segments_for(text):
    normalized = unicodedata.normalize('NFKC', text)
    sentences = [sentence for sentence in SENTENCE.split(normalized) if sentence.strip()]
    words = tokens(normalized)
    fragments = dict.fromkeys(parts(sentences))
    segments = [normalized, *(fragment for fragment in fragments if tokens(fragment) != words)]
    return normalized, sentences, words, segments


def extract(text, encoder=None):
    normalized, sentences, words, segments = segments_for(text)
    units = []
    for index, segment in enumerate(segments):
        segment_words = tokens(segment)
        item = unit(segment, segment_words)
        if index == 0 or len(segment_words) >= 8:
            item['style'] = style(segment, segment_words, sentences if index == 0 else SENTENCE.split(segment))
        units.append(item)
    if encoder:
        for item, vector in zip(units, encoder.encode(segments), strict=True):
            item['vector'] = vector
    return {
        'normalized': ' '.join(words) or ' '.join(normalized.split()).casefold(),
        'units': units,
    }


def serialize(features):
    return zlib.compress(json.dumps(features, ensure_ascii=False, separators=(',', ':')).encode(), 3)


def prepare_for_scoring(result):
    for item in result['units']:
        item['lex'] = [set(values) for values in item['lex']]
        item['pairs'] = set(item['pairs'])
    if 'vector' in result['units'][0]:
        import numpy as np

        vectors = np.asarray([item['vector'] for item in result['units']], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        result['vectors'] = vectors / np.maximum(norms, 1)
    return result


def deserialize(data):
    return prepare_for_scoring(json.loads(zlib.decompress(data)))


def sparse_cosine(left, right):
    if not left or not right:
        return 0.0
    dot = sum(math.sqrt(count * right.get(key, 0)) for key, count in left.items())
    return min(1.0, dot / math.sqrt(sum(left.values()) * sum(right.values())))


def overlap(left, right):
    return len(left & right) / min(len(left), len(right)) if left and right else 0.0


def wording(left, right):
    if left['raw'] == right['raw']:
        return 1.0
    left_sets, right_sets = left['lex'], right['lex']
    return sum(weight * overlap(a, b) for weight, a, b in zip(
        (0.13, 0.24, 0.29, 0.34), left_sets, right_sets, strict=True
    ))


def meaning(left, right, dense=None):
    if left['raw'] == right['raw']:
        return 1.0
    lexical = sparse_cosine(left['concept'], right['concept'])
    ordered = overlap(left['pairs'], right['pairs'])
    similarity = lexical * (0.84 + 0.16 * ordered)
    if dense is not None:
        similarity = max(similarity, max(0.0, (dense - 0.35) / 0.65))
    if bool(left['neg']) != bool(right['neg']):
        similarity *= 0.55
    if left['numbers'] and right['numbers'] and left['numbers'] != right['numbers']:
        similarity *= 0.78
    return similarity


def style_similarity(left, right):
    distance = sum(abs(x - y) for x, y in zip(left, right, strict=True))
    return math.exp(-0.46 * distance)


def compare(current, previous, scores):
    a, b = current['units'][0], previous['units'][0]
    scores[0] = max(scores[0], wording(a, b))
    scores[1] = max(scores[1], sparse_cosine(a['topic'], b['topic']))
    dense = (current['vectors'] @ previous['vectors'].T
             if 'vectors' in current and 'vectors' in previous else None)
    scores[2] = max(scores[2], meaning(a, b, float(dense[0, 0]) if dense is not None else None))
    scores[3] = max(scores[3], style_similarity(a['style'], b['style']))
    if min(scores) >= 0.95:
        return
    left_units = [(i, item) for i, item in enumerate(current['units']) if item['n'] >= 3]
    right_units = [(i, item) for i, item in enumerate(previous['units']) if item['n'] >= 3]
    for i, left in left_units:
        for j, right in right_units:
            if not (i or j):
                continue
            confidence = min(0.95, 0.64 + min(left['n'], right['n']) * 0.04)
            if scores[0] < confidence:
                scores[0] = max(scores[0], wording(left, right) * confidence)
            if scores[1] < confidence:
                scores[1] = max(scores[1], sparse_cosine(left['topic'], right['topic']) * confidence)
            if scores[2] < confidence:
                cosine = float(dense[i, j]) if dense is not None else None
                scores[2] = max(scores[2], meaning(left, right, cosine) * confidence)
            if scores[3] < 0.85 and 'style' in left and 'style' in right:
                scores[3] = max(scores[3], style_similarity(left['style'], right['style']) * 0.85)


def style_keys(values, neighbors):
    for start in range(0, len(values), 2):
        first, second = [int(value / 0.3) for value in values[start:start + 2]]
        offsets = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)) if neighbors else ((0, 0),)
        for x, y in offsets:
            yield stable_hash(f'{start}:{first + x}:{second + y}')


def vector_keys(vector):
    # Fixed sparse random projections give reproducible cosine LSH bands.
    for band, planes in enumerate(projections(len(vector))):
        bits = sum(
            (sum(vector[axis] * sign for axis, sign in plane) >= 0) << position
            for position, plane in enumerate(planes)
        )
        yield stable_hash(f'emb:{band}:{bits}')


@lru_cache(maxsize=2)
def projections(dimension):
    import random
    rng = random.Random(17039)
    return [
        [
            [(axis, rng.choice((-1, 1))) for axis in rng.sample(range(dimension), min(32, dimension))]
            for _ in range(8)
        ]
        for _ in range(12)
    ]


def index_groups(features, query=False):
    whole, *fragments = features['units']
    yield 4, (whole['raw'],)
    for index, item in enumerate((whole, *fragments)):
        # Bounded probes per fragment, with reserved keys for exact phrase matches.
        budget = 18 if index == 0 else 8
        _, pairs, triples, chars = item['lex']
        yield 0, tuple(dict.fromkeys([item['raw'], *triples[:budget], *pairs[:budget // 2],
                                     *chars[:budget // 2]]))
        yield 1, tuple(sorted(stable_hash(word) for word in item['topic'])[:budget])
        yield 2, tuple(sorted(stable_hash(word) for word in item['concept'])[:budget])
        if 'style' in item:
            yield 3, tuple(style_keys(item['style'], query and index == 0))
        if 'vector' in item:
            yield 5, tuple(vector_keys(item['vector']))


def report(percentages):
    return '\n'.join(f'Уникальность {label}: {score}%' for label, score in zip(
        LABELS, percentages, strict=True))
