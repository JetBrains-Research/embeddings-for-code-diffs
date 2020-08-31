import os
import sys
from typing import Tuple

from torchtext import data
from torchtext.data import Field, Dataset

from datasets.dataset_utils import create_filter_predicate_on_length
from datasets.HunkSplitter import HunkSplitter
from edit_representation.sequence_encoding.Differ import Differ
from neural_editor.seq2seq.config import Config


class StablePatchPredictionDataset(data.Dataset):
    """Defines a dataset for commit message generation. It parses text files with tokens"""

    def __init__(self, path: str, diffs_field: Field, config: Config, filter_pred, max_size=None) -> None:
        fields = [('src', diffs_field), ('trg', Field(sequential=False, use_vocab=False)),
                  ('diff_alignment', diffs_field), ('diff_prev', diffs_field), ('diff_updated', diffs_field),
                  ('updated', diffs_field),
                  ('ids', Field(sequential=False, use_vocab=False))]
        examples = []
        differ = Differ(config['REPLACEMENT_TOKEN'], config['DELETION_TOKEN'],
                        config['ADDITION_TOKEN'], config['UNCHANGED_TOKEN'],
                        config['PADDING_TOKEN'])
        hunk_splitter = HunkSplitter(config['CONTEXT_SIZE_FOR_HUNKS'], differ, config)
        with open(os.path.join(path, 'trg.txt'), mode='r', encoding='utf-8') as stable, \
                open(os.path.join(path, 'prev.txt'), mode='r', encoding='utf-8') as prev, \
                open(os.path.join(path, 'updated.txt'), mode='r', encoding='utf-8') as updated:
            total = 0
            errors = 0
            for stable_line, prev_line, updated_line in zip(stable, prev, updated):
                total += 1
                if max_size is not None and total > max_size:
                    break

                stable_line, prev_line, updated_line = stable_line.strip(), prev_line.strip(), updated_line.strip()
                diff, prev_line, updated_line = hunk_splitter.diff_sequences_and_add_hunks(prev_line, updated_line)
                is_correct, error = filter_pred((prev_line.split(' '), updated_line.split(' '),
                                                 diff[0], diff[1], diff[2]))
                if not is_correct:
                    errors += 1
                    print(f'Incorrect {total - 1} example is seen. Error: {error}', file=sys.stderr)
                    continue
                examples.append(data.Example.fromlist(
                    [prev_line, int(stable_line), diff[0], diff[1], diff[2], updated_line, len(examples)], fields))
            print(f'Errors: {errors} / {total - 1} = {errors / total}')
        super(StablePatchPredictionDataset, self).__init__(examples, fields)

    @staticmethod
    def load_data(diffs_field: Field, verbose: bool, config: Config) -> Tuple[Dataset, Dataset, Dataset]:
        filter_predicate = create_filter_predicate_on_length(config['TOKENS_CODE_CHUNK_MAX_LEN'])
        train_data = StablePatchPredictionDataset(os.path.join(config['DATASET_ROOT_COMMIT'], 'train'),
                                                  diffs_field, config, filter_pred=filter_predicate,
                                                  max_size=config['MAX_NUMBER_OF_EXAMPLES_TEST'])
        val_data = StablePatchPredictionDataset(os.path.join(config['DATASET_ROOT_COMMIT'], 'val'),
                                                diffs_field, config, filter_pred=filter_predicate,
                                                max_size=config['MAX_NUMBER_OF_EXAMPLES_TEST'])
        test_data = StablePatchPredictionDataset(os.path.join(config['DATASET_ROOT_COMMIT'], 'test'),
                                                 diffs_field, config, filter_pred=filter_predicate,
                                                 max_size=config['MAX_NUMBER_OF_EXAMPLES_TEST'])
        if verbose:
            StablePatchPredictionDataset.print_data_info(train_data, val_data, test_data, diffs_field, config)
        return train_data, val_data, test_data

    @staticmethod
    def print_data_info(train_data: Dataset, valid_data: Dataset, test_data: Dataset,
                        diffs_field: Field, config: Config) -> None:
        """ This prints some useful stuff about our data sets. """

        print("Data set sizes (number of sentence pairs):")
        print('train', len(train_data))
        print('valid', len(valid_data))
        print('test', len(test_data), "\n")

        max_src_len = max(len(example.src) for dataset in (train_data, valid_data, test_data) for example in dataset)
        max_diff_len = max(len(example.diff_alignment) for dataset
                           in (train_data, valid_data, test_data) for example in dataset)
        print(f'Max src sequence length in tokens: {max_src_len}')
        print(f'Max diff sequence length in tokens: {max_diff_len}', '\n')

        print("First training example:")
        print("src           :", " ".join(vars(train_data[0])['src']))
        print("updated       :", " ".join(vars(train_data[0])['updated']))
        print("diff_alignment:", " ".join(vars(train_data[0])['diff_alignment']))
        print("diff_prev     :", " ".join(vars(train_data[0])['diff_prev']))
        print("diff_updated  :", " ".join(vars(train_data[0])['diff_updated']))
        print("trg           :", vars(train_data[0])['trg'])

        print("Most common words in diffs_field vocabulary:")
        print("\n".join(["%10s %10d" % x for x in diffs_field.vocab.freqs.most_common(10)]), "\n")

        print("First 10 words in diffs_field vocabulary:")
        print("\n".join('%02d %s' % (i, t) for i, t in enumerate(diffs_field.vocab.itos[:10])), "\n")

        special_tokens = [config['UNK_TOKEN'], config['PAD_TOKEN'], config['SOS_TOKEN'], config['EOS_TOKEN'],
                          config['REPLACEMENT_TOKEN'], config['DELETION_TOKEN'], config['ADDITION_TOKEN'],
                          config['UNCHANGED_TOKEN'], config['PADDING_TOKEN']]
        print("Special words frequency and ids in diffs_field vocabulary: ")
        for special_token in special_tokens:
            print(f"{special_token} {diffs_field.vocab.freqs[special_token]} {diffs_field.vocab.stoi[special_token]}")

        print("Number of words (types) in diffs_field vocabulary:", len(diffs_field.vocab))
