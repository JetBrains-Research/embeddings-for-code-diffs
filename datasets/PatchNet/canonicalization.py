import sys
import time
from datetime import timedelta
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Any, Tuple, List

from pygments.token import Token

from datasets.PatchNet.PatchNetDataset import PatchNetDataset, DataSample

DEFAULT_MIN_FREQ = 100

MIN_FREQ_THRESHOLDS = {
    Token.Name: DEFAULT_MIN_FREQ,
    Token.Operator: DEFAULT_MIN_FREQ,
    Token.Punctuation: DEFAULT_MIN_FREQ,
    Token.Literal.String: DEFAULT_MIN_FREQ,
    Token.Keyword: DEFAULT_MIN_FREQ,
    Token.Name.Builtin: DEFAULT_MIN_FREQ,
    Token.Keyword.Type: DEFAULT_MIN_FREQ,
    Token.Name.Function: DEFAULT_MIN_FREQ,
    Token.Keyword.Reserved: DEFAULT_MIN_FREQ,
    Token.Literal.String.Escape: DEFAULT_MIN_FREQ,
    Token.Literal.String.Affix: DEFAULT_MIN_FREQ,
    Token.Literal.Number.Integer: DEFAULT_MIN_FREQ,
    Token.Name.Label: DEFAULT_MIN_FREQ,
    Token.Literal.Number.Hex: DEFAULT_MIN_FREQ,
    Token.Comment.Preproc: DEFAULT_MIN_FREQ,
    Token.Comment.PreprocFile: DEFAULT_MIN_FREQ,
    Token.Literal.Number.Oct: DEFAULT_MIN_FREQ,
    Token.Error: DEFAULT_MIN_FREQ,
    Token.Literal.String.Char: DEFAULT_MIN_FREQ,
    Token.Literal.Number.Float: DEFAULT_MIN_FREQ,
    Token.Comment: 1e15,
}


def sort_counter_by_token_types(old_counter: Counter):
    new_counter = defaultdict(lambda: Counter())
    for k, v in old_counter.items():
        new_counter[k[0]][k[1]] = v
    return new_counter


def sort_counter_by_token_texts(old_counter: Counter):
    new_counter = defaultdict(lambda: Counter())
    for k, v in old_counter.items():
        new_counter[k[1]][k[0]] = v
    return new_counter


def get_tokens_to_leave(counter_by_types: Dict[Any, Counter]):
    new_dict = {}
    approximate_vocab_size = 0
    for token_type, counter in counter_by_types.items():
        if token_type not in MIN_FREQ_THRESHOLDS:
            print('Unknown token type!')
            print(token_type)
            print(counter)
            exit(1)
        new_dict[token_type] = Counter({k: v for k, v in counter.items() if v >= MIN_FREQ_THRESHOLDS[token_type]})
        print(f'Size of {token_type} dict: {len(new_dict[token_type])}')
        approximate_vocab_size += len(new_dict[token_type])
    print(f'Approximate vocab size: {approximate_vocab_size}')
    return new_dict


def is_text_allowed(token_text: str):
    banned_substrings = ['\n', '\r\n', '\r', '\t']
    allowed_by_correct_splitting = ['ifdef CONFIG_SMP', 'ifndef __ASSEMBLY__', 'ifdef CONFIG_PM',
                                    'ifdef CONFIG_OF', 'ifdef CONFIG_PM_SLEEP', 'GPL v2', '#if 0']
    if token_text in allowed_by_correct_splitting:
        return True
    for banned_substring in banned_substrings:
        if banned_substring in token_text:
            return False
    return True


def create_mapping(tokens_to_leave, prev, updated) -> Dict[Tuple[Any, str], str]:
    def calculate_mapping(token_sequence):
        for token in token_sequence:
            token_type = token[0]
            token_text = token[1]
            if token in mapping:
                continue
            elif token_text in tokens_to_leave[token_type]:
                mapping[token] = token_text
            else:
                mapping[token] = f'<{token_type}_{numbers_counter[token_type]}>'
                numbers_counter[token_type] += 1
    mapping = {}
    numbers_counter = defaultdict(lambda: 0)
    calculate_mapping(prev)
    calculate_mapping(updated)
    return mapping


def check_mapping(mapping):
    for text in mapping.values():
        if not is_text_allowed(text):
            raise Exception(f'Token text "{text}" is not allowed.')


def canonicalize_sample(tokens_to_leave, prev_tokens, updated_tokens) -> \
        Tuple[List[Tuple[Any, str]], List[Tuple[Any, str]]]:
    mapping = create_mapping(tokens_to_leave, prev_tokens, updated_tokens)
    check_mapping(mapping)
    return [(token[0], mapping[token]) for token in prev_tokens], \
           [(token[0], mapping[token]) for token in updated_tokens]


def canonicalize(tokens_to_leave: Dict[Any, Counter], dataset: PatchNetDataset):
    vocab = set()
    lines_prev = []
    lines_updated = []
    start = time.time()
    for idx, sample in enumerate(dataset.data_samples):
        prev, updated = canonicalize_sample(tokens_to_leave, sample.commit.get_prev(), sample.commit.get_updated())
        lines_prev.append(' '.join(token[1] for token in prev))
        lines_updated.append(' '.join(token[1] for token in updated))
        for token in lines_prev[-1].split() + lines_updated[-1].split():
            vocab.add(token)

        if (idx + 1) % 50000 == 0:
            end = time.time()
            duration = end - start
            print(f'Processed {idx + 1} / {len(dataset.data_samples)} samples')
            print(f'Time elapsed: {str(timedelta(seconds=duration))}', flush=True)
            start = end
    print(f'Final vocab size: {len(vocab)}')
    dataset.root.joinpath(f'{DEFAULT_MIN_FREQ}_canonicalized_prev.txt').write_text('\n'.join(lines_prev))
    dataset.root.joinpath(f'{DEFAULT_MIN_FREQ}_canonicalized_updated.txt').write_text('\n'.join(lines_updated))


def perform_canonicalization():
    if len(sys.argv) < 2:
        print('Usage: <root where to save processed data>')
        exit(1)
    root = Path(sys.argv[1])
    patch_net_dataset = PatchNetDataset(root, None, None)
    patch_net_dataset.load()
    token_types_counter = sort_counter_by_token_types(patch_net_dataset.tokens_counter)
    tokens_to_leave = get_tokens_to_leave(token_types_counter)
    print('Starting canonicalization!', flush=True)
    canonicalize(tokens_to_leave, patch_net_dataset)


if __name__ == "__main__":
    perform_canonicalization()
