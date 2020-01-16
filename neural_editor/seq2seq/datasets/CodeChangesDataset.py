import os

from torchtext import data
from torchtext.data import Field

from edit_representation.sequence_encoding.Differ import Differ
from neural_editor.seq2seq.config import Config


class CodeChangesTokensDataset(data.Dataset):
    """Defines a dataset for code changes. It parses text files with tokens"""

    def __init__(self, path: str, field: Field, config: Config, **kwargs) -> None:
        """Create a TranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            fields: A field that will be used for data.
            prefix: file prefix (train, val or test)
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('src', field), ('trg', field),
                  ('diff_alignment', field), ('diff_prev', field), ('diff_updated', field)]
        examples = []
        differ = Differ(config['REPLACEMENT_TOKEN'], config['DELETION_TOKEN'],
                        config['ADDITION_TOKEN'], config['UNCHANGED_TOKEN'],
                        config['PADDING_TOKEN'])
        with open(os.path.join(path, 'prev.txt'), mode='r', encoding='utf-8') as prev, \
                open(os.path.join(path, 'updated.txt'), mode='r', encoding='utf-8') as updated:
            for prev_line, updated_line in zip(prev, updated):
                prev_line, updated_line = prev_line.strip(), updated_line.strip()
                if prev_line != '' and updated_line != '':
                    diff = differ.diff_tokens_fast_lvn(prev_line.split(' '), updated_line.split(' '),
                                                       leave_only_changed=config['LEAVE_ONLY_CHANGED'])
                    examples.append(data.Example.fromlist(
                        [prev_line, updated_line, diff[0], diff[1], diff[2]], fields))
        super(CodeChangesTokensDataset, self).__init__(examples, fields, **kwargs)
