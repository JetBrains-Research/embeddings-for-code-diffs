import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import List
from tqdm.auto import tqdm

IDENTIFIER_TOKEN = '<IDENTIFIER>'


def build_vocab(lines: List[str]):
    return Counter([token for line in lines for token in line.split()])


def remove_identifiers_from_lines(lines, all_identifiers, identifiers_to_leave):
    new_lines = []
    for line in tqdm(lines):
        new_line = [token for token in line.split() if token in identifiers_to_leave or token not in all_identifiers]
        new_lines.append(' '.join(new_line))
    return new_lines


def replace_identifiers_from_lines(lines, all_identifiers, identifiers_to_leave):
    new_lines = []
    for line in tqdm(lines):
        new_line = [token if token in identifiers_to_leave or token not in all_identifiers else IDENTIFIER_TOKEN
                    for token in line.split()]
        new_lines.append(' '.join(new_line))
    return new_lines


def remove_identifiers():
    if len(sys.argv) != 3:
        print('Usage: <root where to save processed data> <n most common to leave>')
        exit(1)
    root = Path(sys.argv[1])
    n = int(sys.argv[2])
    counter: Counter = pickle.load(root.joinpath('identifier_names_counter.pkl').open('rb'))
    prev_lines = root.joinpath('prev.txt').read_text().splitlines(keepends=False)
    updated_lines = root.joinpath('updated.txt').read_text().splitlines(keepends=False)
    vocab = build_vocab(prev_lines + updated_lines)
    most_common = counter.most_common(n)
    tokens_to_leave = set([el[0] for el in most_common])
    print(f'Max len in tokens: {max((max(len(prev.split()), len(updated.split())) for prev, updated in zip(prev_lines, updated_lines)))}')
    print(f'Vocab size: {len(vocab)}')
    print(f'Counter size: {len(counter)}')
    print(f'Counter {n} most common freq: {most_common[-1][1]}')
    new_prev_lines = remove_identifiers_from_lines(prev_lines, set(counter.keys()), tokens_to_leave)
    new_updated_lines = remove_identifiers_from_lines(updated_lines, set(counter.keys()), tokens_to_leave)
    root.joinpath('removed_identifier_prev.txt').write_text('\n'.join(new_prev_lines))
    root.joinpath('removed_identifier_updated.txt').write_text('\n'.join(new_updated_lines))


def replace_identifiers():
    if len(sys.argv) != 3:
        print('Usage: <root where to save processed data> <min frequency to leave>')
        exit(1)
    root = Path(sys.argv[1])
    min_freq = int(sys.argv[2])
    counter: Counter = pickle.load(root.joinpath('identifier_names_counter.pkl').open('rb'))
    prev_lines = root.joinpath('prev.txt').read_text().splitlines(keepends=False)
    updated_lines = root.joinpath('updated.txt').read_text().splitlines(keepends=False)
    vocab = build_vocab(prev_lines + updated_lines)
    tokens_to_leave = set([el[0] for el in counter.items() if el[1] >= min_freq])
    print(f'Max len in tokens: {max((max(len(prev.split()), len(updated.split())) for prev, updated in zip(prev_lines, updated_lines)))}')
    print(f'Vocab size: {len(vocab)}')
    print(f'Counter size: {len(counter)}')
    print(f'Counter size of tokens to leave: {len(tokens_to_leave)}')
    new_prev_lines = replace_identifiers_from_lines(prev_lines, set(counter.keys()), tokens_to_leave)
    new_updated_lines = replace_identifiers_from_lines(updated_lines, set(counter.keys()), tokens_to_leave)
    root.joinpath('replaced_identifier_prev.txt').write_text('\n'.join(new_prev_lines))
    root.joinpath('replaced_identifier_updated.txt').write_text('\n'.join(new_updated_lines))


def remove_empty():
    if len(sys.argv) != 2:
        print('Usage: <root where to save processed data>')
        exit(1)
    root = Path(sys.argv[1])
    filenames = ['prev.txt', 'updated.txt', 'trg.txt', 'ids.txt']

    data = list(zip(*[root.joinpath(filename).read_text().splitlines(keepends=False) for filename in filenames]))
    data = [el for el in data if el[0] != '' and el[1] != '']
    filenames_lines = {filename: [] for filename in filenames}
    for data_sample in data:
        for i, filename in enumerate(filenames_lines):
            filenames_lines[filename].append(data_sample[i])
    for filename, lines in filenames_lines.items():
        root.joinpath(filename).write_text('\n'.join(lines))


if __name__ == "__main__":
    remove_identifiers()
