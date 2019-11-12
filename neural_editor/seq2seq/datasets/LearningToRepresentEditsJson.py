import io
import json
import os

from collatex import collate
from torchtext import data
from tqdm import tqdm

from edit_representation.sequence_encoding.Differ import Differ
from neural_editor.seq2seq.train_config import CONFIG


class LearningToRepresentEditsJson(data.Dataset):
    """Defines a dataset for learning to represent edits. It parses json files."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, field, **kwargs):
        """Create a LearningToRepresentEdits dataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            fields: A field that will be used for data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('src', field), ('trg', field)]
        examples = []
        data_filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        for data_filename in data_filenames:
            with io.open(os.path.join(path, data_filename), mode='r', encoding='utf-8-sig') as data_file:
                for line in data_file:
                    diff = json.loads(line)
                    src_tokens, trg_tokens = diff['PrevCodeChunkTokens'], diff['UpdatedCodeChunkTokens']
                    examples.append(data.Example.fromlist([src_tokens, trg_tokens], fields))
        super(LearningToRepresentEditsJson, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, path, field, train='train', validation='val', test='test',  **kwargs):
        """Create dataset objects for splits of a TranslationDataset.

        Arguments:
            field: A fields that will be used for data.
            path (str): A root where data files exist.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        train_data = None if train is None else cls(
            os.path.join(path, train), field, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), field, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), field, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


class LearningToRepresentEditsTokenStrings(data.Dataset):
    """Defines a dataset for learning to represent edits. It parses text files with tokens"""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, prefix, field, **kwargs):
        """Create a TranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            fields: A field that will be used for data.
            prefix: file prefix (train, val or test)
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('src', field), ('trg', field), ('diff_alignment', field), ('diff_prev', field), ('diff_updated', field)]
        examples = []
        differ = Differ(CONFIG['REPLACEMENT_SYMBOL'], CONFIG['DELETION_SYMBOL'],
                        CONFIG['ADDITION_SYMBOL'], CONFIG['UNCHANGED_SYMBOL'],
                        CONFIG['PADDING_SYMBOL'])
        with open(os.path.join(path, f'{prefix}_tokens_prev_text'), mode='r', encoding='utf-8') as prev,\
             open(os.path.join(path, f'{prefix}_tokens_updated_text'), mode='r', encoding='utf-8') as updated:
            for prev_line, updated_line in tqdm(zip(prev, updated)):
                prev_line, updated_line = prev_line.strip(), updated_line.strip()
                if prev_line != '' and updated_line != '':
                    # TODO: change symbols in CONFIG
                    diff = differ.diff_tokens_fast_lvn(prev_line.split(' '), updated_line.split(' '))
                    examples.append(data.Example.fromlist(
                        [prev_line, updated_line, diff[0], diff[1], diff[2]], fields))
        super(LearningToRepresentEditsTokenStrings, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, path, field, train='train', validation='val', test='test',  **kwargs):
        """Create dataset objects for splits of a TranslationDataset.

        Arguments:
            field: A fields that will be used for data.
            path (str): A root where data files exist.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        train_data = None if train is None else cls(
            os.path.join(path, train), train, field, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), validation, field, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), test, field, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)
