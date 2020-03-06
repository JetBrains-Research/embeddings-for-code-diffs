import os
import sys
from typing import Tuple

from torchtext import data
from torchtext.data import Field, Dataset

from datasets.dataset_utils import create_filter_predicate_on_length, create_filter_predicate_on_code_and_msg
from edit_representation.sequence_encoding.Differ import Differ
from neural_editor.seq2seq.config import Config


class CommitMessageGenerationDataset(data.Dataset):
    """Defines a dataset for commit message generation. It parses text files with tokens"""

    def __init__(self, path: str, src_field: Field, trg_field: Field, diffs_field: Field,
                 config: Config, filter_pred) -> None:
        """Create a TranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            fields: A field that will be used for data.
            prefix: file prefix (train, val or test)
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('src', src_field), ('trg', trg_field),
                  ('diff_alignment', diffs_field), ('diff_prev', diffs_field), ('diff_updated', diffs_field),
                  ('ids', Field(sequential=False, use_vocab=False))]
        examples = []
        differ = Differ(config['REPLACEMENT_TOKEN'], config['DELETION_TOKEN'],
                        config['ADDITION_TOKEN'], config['UNCHANGED_TOKEN'],
                        config['PADDING_TOKEN'])
        with open(os.path.join(path, 'diff.txt'), mode='r', encoding='utf-8') as diff, \
                open(os.path.join(path, 'msg.txt'), mode='r', encoding='utf-8') as msg, \
                open(os.path.join(path, 'prev.txt'), mode='r', encoding='utf-8') as prev, \
                open(os.path.join(path, 'updated.txt'), mode='r', encoding='utf-8') as updated:
            for diff_line, msg_line, prev_line, updated_line in zip(diff, msg, prev, updated):
                diff_line, msg_line, prev_line, updated_line = \
                    diff_line.strip(), msg_line.strip(), prev_line.strip(), updated_line.strip()
                diff = differ.diff_tokens_fast_lvn(prev_line.split(' '), updated_line.split(' '),
                                                   leave_only_changed=config['LEAVE_ONLY_CHANGED'])
                is_correct, error = filter_pred((diff_line.split(' '), msg_line.split(' '),
                                                 diff[0], diff[1], diff[2]))
                if not is_correct:
                    print(f'Incorrect example is seen. Error: {error}', file=sys.stderr)
                    continue
                examples.append(data.Example.fromlist(
                    [diff_line, msg_line, diff[0], diff[1], diff[2], len(examples)], fields))
        super(CommitMessageGenerationDataset, self).__init__(examples, fields)

    @staticmethod
    def load_data(diffs_field: Field,
                  verbose: bool, config: Config) -> Tuple[Dataset, Dataset, Dataset, Tuple[Field, Field, Field]]:
        # TODO: is src_field = diffs_field?
        src_field: Field = data.Field(batch_first=True, lower=config['LOWER'], include_lengths=True,
                                      unk_token=config['UNK_TOKEN'], pad_token=config['PAD_TOKEN'],
                                      init_token=config['SOS_TOKEN'],
                                      eos_token=config['EOS_TOKEN'])
        trg_field: Field = data.Field(batch_first=True, lower=config['LOWER_COMMIT_MSG'], include_lengths=True,
                                      unk_token=config['UNK_TOKEN'], pad_token=config['PAD_TOKEN'],
                                      init_token=config['SOS_TOKEN'],
                                      eos_token=config['EOS_TOKEN'])

        filter_predicate = create_filter_predicate_on_code_and_msg(config['TOKENS_CODE_CHUNK_MAX_LEN'], config['MSG_MAX_LEN'])
        train_data = CommitMessageGenerationDataset(os.path.join(config['DATASET_ROOT_COMMIT'], 'train'),
                                                    src_field, trg_field, diffs_field,
                                                    config, filter_pred=filter_predicate)
        val_data = CommitMessageGenerationDataset(os.path.join(config['DATASET_ROOT_COMMIT'], 'val'),
                                                  src_field, trg_field, diffs_field,
                                                  config, filter_pred=filter_predicate)
        test_data = CommitMessageGenerationDataset(os.path.join(config['DATASET_ROOT_COMMIT'], 'test'),
                                                   src_field, trg_field, diffs_field,
                                                   config, filter_pred=filter_predicate)
        src_field.build_vocab(train_data.src, min_freq=config['TOKEN_MIN_FREQ'])
        trg_field.build_vocab(train_data.trg, min_freq=config['TOKEN_MIN_FREQ'])
        if verbose:
            CommitMessageGenerationDataset.print_data_info(train_data, val_data, test_data, src_field, trg_field, diffs_field, config)
        return train_data, val_data, test_data, (src_field, trg_field, diffs_field)

    @staticmethod
    def print_data_info(train_data: Dataset, valid_data: Dataset, test_data: Dataset,
                        src_field: Field, trg_field: Field, diffs_field: Field,
                        config: Config) -> None:
        """ This prints some useful stuff about our data sets. """

        print("Data set sizes (number of sentence pairs):")
        print('train', len(train_data))
        print('valid', len(valid_data))
        print('test', len(test_data), "\n")

        max_src_len = max(len(example.src) for dataset in (train_data, valid_data, test_data) for example in dataset)
        max_trg_len = max(len(example.trg) for dataset in (train_data, valid_data, test_data) for example in dataset)
        max_diff_len = max(len(example.diff_alignment) for dataset
                           in (train_data, valid_data, test_data) for example in dataset)
        print(f'Max src sequence length in tokens: {max_src_len}')
        print(f'Max trg sequence length in tokens: {max_trg_len}')
        print(f'Max diff sequence length in tokens: {max_diff_len}', '\n')

        print("First training example:")
        print("src:", " ".join(vars(train_data[0])['src']))
        print("trg:", " ".join(vars(train_data[0])['trg']))
        print("diff_alignment:", " ".join(vars(train_data[0])['diff_alignment']))
        print("diff_prev:", " ".join(vars(train_data[0])['diff_prev']))
        print("diff_updated:", " ".join(vars(train_data[0])['diff_updated']))

        print("Most common words in src vocabulary:")
        print("\n".join(["%10s %10d" % x for x in src_field.vocab.freqs.most_common(10)]), "\n")
        print()
        print("Most common words in trg vocabulary:")
        print("\n".join(["%10s %10d" % x for x in trg_field.vocab.freqs.most_common(10)]), "\n")
        print()
        print("Most common words in diff vocabulary:")
        print("\n".join(["%10s %10d" % x for x in diffs_field.vocab.freqs.most_common(10)]), "\n")

        print("First 10 words in src vocabulary:")
        print("\n".join('%02d %s' % (i, t) for i, t in enumerate(src_field.vocab.itos[:10])), "\n")
        print()
        print("First 10 words in trg vocabulary:")
        print("\n".join('%02d %s' % (i, t) for i, t in enumerate(trg_field.vocab.itos[:10])), "\n")
        print()
        print("First 10 words in diff vocabulary:")
        print("\n".join('%02d %s' % (i, t) for i, t in enumerate(diffs_field.vocab.itos[:10])), "\n")

        special_tokens = [config['UNK_TOKEN'], config['PAD_TOKEN'], config['SOS_TOKEN'], config['EOS_TOKEN']]
        print("Special words frequency and ids in src vocabulary: ")
        for special_token in special_tokens:
            print(f"{special_token} {src_field.vocab.freqs[special_token]} {src_field.vocab.stoi[special_token]}")
        print("Special words frequency and ids in trg vocabulary: ")
        for special_token in special_tokens:
            print(f"{special_token} {trg_field.vocab.freqs[special_token]} {trg_field.vocab.stoi[special_token]}")
        special_tokens = [config['UNK_TOKEN'], config['PAD_TOKEN'], config['SOS_TOKEN'], config['EOS_TOKEN'],
                          config['REPLACEMENT_TOKEN'], config['DELETION_TOKEN'], config['ADDITION_TOKEN'],
                          config['UNCHANGED_TOKEN'], config['PADDING_TOKEN']]
        print("Special words frequency and ids in diffs_field vocabulary: ")
        for special_token in special_tokens:
            print(f"{special_token} {diffs_field.vocab.freqs[special_token]} {diffs_field.vocab.stoi[special_token]}")

        print("Number of words (types) in src vocabulary:", len(src_field.vocab))
        print("Number of words (types) in trg vocabulary:", len(trg_field.vocab))
        print("Number of words (types) in diff vocabulary:", len(diffs_field.vocab))
