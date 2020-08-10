import os
from typing import Tuple

from torch.utils.data import Dataset
from torchtext import data
from torchtext.data import Field, Dataset

from datasets.HunkSplitter import HunkSplitter
from datasets.dataset_utils import create_filter_predicate_on_length
from edit_representation.sequence_encoding.Differ import Differ
from neural_editor.seq2seq.config import Config


class CodeChangesTokensDataset(data.Dataset):
    """Defines a dataset for code changes. It parses text files with tokens"""

    def __init__(self, path: str, field: Field, config: Config, filter_pred, max_size=None) -> None:
        """Create a TranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            fields: A field that will be used for data.
            prefix: file prefix (train, val or test)
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('src', field), ('trg', field),
                  ('diff_alignment', field), ('diff_prev', field), ('diff_updated', field),
                  ('stable', Field(sequential=False, use_vocab=False)),
                  ('ids', Field(sequential=False, use_vocab=False)),
                  ('original_ids', Field(sequential=False, use_vocab=False))]
        examples = []
        differ = Differ(config['REPLACEMENT_TOKEN'], config['DELETION_TOKEN'],
                        config['ADDITION_TOKEN'], config['UNCHANGED_TOKEN'],
                        config['PADDING_TOKEN'])
        hunk_splitter = HunkSplitter(config['CONTEXT_SIZE_FOR_HUNKS'], differ, config)
        with open(os.path.join(path, 'prev.txt'), mode='r', encoding='utf-8') as prev, \
                open(os.path.join(path, 'updated.txt'), mode='r', encoding='utf-8') as updated, \
                open(os.path.join(path, 'trg.txt'), mode='r', encoding='utf-8') as stable, \
                open(os.path.join(path, 'ids.txt'), mode='r', encoding='utf-8') as original_ids:
            total = 0
            errors = 0
            for prev_line, updated_line, stable_line, original_id_line in zip(prev, updated, stable, original_ids):
                total += 1
                if max_size is not None and total > max_size:
                    break

                prev_line, updated_line, stable_line, original_id_line = \
                    prev_line.strip(), updated_line.strip(), stable_line.strip(), original_id_line.strip()
                diff, prev_line, updated_line = hunk_splitter.diff_sequences_and_add_hunks(prev_line, updated_line)
                is_correct, error = filter_pred((prev_line.split(' '), updated_line.split(' '),
                                                 diff[0], diff[1], diff[2]))
                if not is_correct:
                    errors += 1
                    # print(f'Incorrect example is seen. Error: {error}', file=sys.stderr)
                    continue
                examples.append(data.Example.fromlist(
                    [prev_line, updated_line, diff[0], diff[1], diff[2], int(stable_line),
                     len(examples), int(original_id_line)], fields))
        print(f'Errors: {errors} / {total} = {errors / total}')
        super(CodeChangesTokensDataset, self).__init__(examples, fields)

    @staticmethod
    def load_datasets(path: str, field: Field, config: Config,
                      train: str = 'train', val: str = 'val', test: str = 'test') -> Tuple[Dataset, Dataset, Dataset]:
        filter_predicate = create_filter_predicate_on_length(config['TOKENS_CODE_CHUNK_MAX_LEN'])
        train_data: Dataset = CodeChangesTokensDataset(os.path.join(path, train), field, config, filter_predicate,
                                                       max_size=config['MAX_NUMBER_OF_EXAMPLES_TRAIN'])
        val_data: Dataset = CodeChangesTokensDataset(os.path.join(path, val), field, config, filter_predicate,
                                                     max_size=config['MAX_NUMBER_OF_EXAMPLES_VAL'])
        test_data: Dataset = CodeChangesTokensDataset(os.path.join(path, test), field, config, filter_predicate,
                                                      max_size=config['MAX_NUMBER_OF_EXAMPLES_TEST'])
        return train_data, val_data, test_data

    @staticmethod
    def load_data(verbose: bool, config: Config) -> Tuple[Dataset, Dataset, Dataset, Field]:
        diffs_field: Field = data.Field(batch_first=True, lower=config['LOWER'], include_lengths=True,
                                        unk_token=config['UNK_TOKEN'], pad_token=config['PAD_TOKEN'],
                                        init_token=config['SOS_TOKEN'],
                                        eos_token=config['EOS_TOKEN'])

        train_data, val_data, test_data = CodeChangesTokensDataset.load_datasets(config['DATASET_ROOT'], diffs_field,
                                                                                 config)
        diffs_field.build_vocab(train_data.src, train_data.trg,
                                train_data.diff_alignment, train_data.diff_prev,
                                train_data.diff_updated, min_freq=config['TOKEN_MIN_FREQ'])
        if verbose:
            CodeChangesTokensDataset.print_data_info(train_data, val_data, test_data, diffs_field, config)
        return train_data, val_data, test_data, diffs_field

    @staticmethod
    def print_data_info(train_data: Dataset, valid_data: Dataset, test_data: Dataset, field: Field,
                        config: Config) -> None:
        """ This prints some useful stuff about our data sets. """

        print("Data set sizes (number of sentence pairs):")
        print('train', len(train_data))
        print('valid', len(valid_data))
        print('test', len(test_data), "\n")

        max_src_trg_len = max(max(len(example.src), len(example.trg)) for dataset
                              in (train_data, valid_data, test_data) for example in dataset)
        max_diff_len = max(len(example.diff_alignment) for dataset
                           in (train_data, valid_data, test_data) for example in dataset)
        print(f'Max src or trg sequence length in tokens: {max_src_trg_len}')
        print(f'Max diff sequence length in tokens: {max_diff_len}', '\n')

        print("First training example:")
        print("src           :", " ".join(vars(train_data[0])['src']))
        print("trg           :", " ".join(vars(train_data[0])['trg']))
        print("diff_alignment:", " ".join(vars(train_data[0])['diff_alignment']))
        print("diff_prev     :", " ".join(vars(train_data[0])['diff_prev']))
        print("diff_updated  :", " ".join(vars(train_data[0])['diff_updated']))
        print("stable        :", vars(train_data[0])['stable'], '\n')

        print("Most common words:")
        print("\n".join(["%10s %10d" % x for x in field.vocab.freqs.most_common(10)]), "\n")

        print("First 10 words:")
        print("\n".join(
            '%02d %s' % (i, t) for i, t in enumerate(field.vocab.itos[:10])), "\n")

        print("Special words frequency and ids: ")
        special_tokens = [config['UNK_TOKEN'], config['PAD_TOKEN'], config['SOS_TOKEN'], config['EOS_TOKEN'],
                          config['REPLACEMENT_TOKEN'], config['DELETION_TOKEN'], config['ADDITION_TOKEN'],
                          config['UNCHANGED_TOKEN'], config['PADDING_TOKEN']]
        for special_token in special_tokens:
            print(f"{special_token} {field.vocab.freqs[special_token]} {field.vocab.stoi[special_token]}")

        print("Number of words (types):", len(field.vocab))
