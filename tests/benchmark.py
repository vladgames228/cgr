import argparse
import random
import resource
import tempfile
from pathlib import Path
from time import perf_counter

from uniqueness.storage import Archive

TOPICS = (
    'На станции ремонтируют пути. График движения меняется до конца недели.',
    'В ботаническом саду распустились тюльпаны. Экскурсии проходят в первой половине дня.',
    'Команда представила новую версию приложения. В редактор добавили историю изменений.',
    'Первая партия книг поступила на склад. Читатели могут забрать заказы в выходные.',
    'В центре города начали замену труб. Работы затронут несколько соседних кварталов.',
    'Хоккейный клуб выиграл домашний матч. Болельщики встретили команду у арены.',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--posts', type=int, default=1500)
    parser.add_argument('--model-dir')
    args = parser.parse_args()
    rng = random.Random(70)
    encoder = None
    if args.model_dir:
        from uniqueness.encoder import Encoder
        encoder = Encoder(args.model_dir)
    with tempfile.TemporaryDirectory() as folder:
        db_path = Path(folder) / 'archive.sqlite3'
        archive = Archive(db_path, encoder.identity if encoder else 'classic-v1', encoder)
        archive.register(1, 2, 3)
        start = perf_counter()
        for identifier in range(1, args.posts + 1):
            pieces = rng.sample(TOPICS, rng.randint(1, 3))
            text = ' '.join(pieces) + f' Подробности в выпуске №{identifier}.'
            archive.index(1, identifier, '2025-01-01T00:00:00+00:00', text)
        elapsed = perf_counter() - start
        archive.close()
        print(f'{args.posts} posts in {elapsed:.2f}s, {args.posts / elapsed:.1f} posts/s')
        print(f'Peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MiB')
        print(f'SQLite {db_path.stat().st_size / 1024**2:.1f} MiB')


if __name__ == '__main__':
    main()
