from torchtext import data
from torchtext.data import Field
import numpy as np


class MatrixDataset(data.Dataset):
    def __init__(self, dataset: data.Dataset, diff_example_ids: np.ndarray, field: Field, **kwargs) -> None:
        fields = [('src', field), ('trg', field),
                  ('diff_alignment', field), ('diff_prev', field), ('diff_updated', field),
                  ('ids', Field(sequential=False, use_vocab=False)),
                  ('src_example_ids', Field(sequential=False, use_vocab=False)),
                  ('diff_example_ids', Field(sequential=False, use_vocab=False))]
        examples = []
        for i, src_example in enumerate(dataset.examples):
            for j, diff_example_id in enumerate(diff_example_ids[i]):
                diff_example = dataset.examples[diff_example_id]
                idx = i * len(dataset) + j
                examples.append(data.Example.fromlist(
                    [src_example.src, src_example.trg,
                     diff_example.diff_alignment, diff_example.diff_prev, diff_example.diff_updated,
                     idx, src_example.ids, diff_example.ids], fields))
        super(MatrixDataset, self).__init__(examples, fields, **kwargs)
