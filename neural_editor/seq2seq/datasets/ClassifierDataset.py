import random

from torchtext import data
from torchtext.data import Field
import numpy as np


class ClassifierDataset(data.Dataset):
    def __init__(self, dataset: data.Dataset, src_example_ids: np.ndarray, diff_example_ids: np.ndarray,
                 field: Field, matrix: np.ndarray, **kwargs) -> None:
        fields = [('original_src', field), ('edit_src', field),
                  ('trg', Field(sequential=False, use_vocab=False)),
                  ('src_example_ids', Field(sequential=False, use_vocab=False)),
                  ('diff_example_ids', Field(sequential=False, use_vocab=False))]
        negative_examples = []
        positive_examples = []
        for src_example_id in src_example_ids:
            src_example = dataset.examples[src_example_id]
            for j, diff_example_id in enumerate(diff_example_ids[src_example_id]):
                diff_example = dataset.examples[diff_example_id]
                trg = matrix[src_example_id][j]
                example = data.Example.fromlist([src_example.src, diff_example.src, trg,
                                                 src_example.ids, diff_example.ids], fields)
                if trg:
                    positive_examples.append(example)
                else:
                    negative_examples.append(example)
        random.shuffle(positive_examples)
        random.shuffle(negative_examples)
        examples = positive_examples[:min(len(positive_examples), len(negative_examples))] + \
                   negative_examples[:min(len(positive_examples), len(negative_examples))]
        random.shuffle(examples)
        super(ClassifierDataset, self).__init__(examples, fields, **kwargs)
