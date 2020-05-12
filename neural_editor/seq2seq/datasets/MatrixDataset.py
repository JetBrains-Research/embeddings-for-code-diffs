from torchtext import data
from torchtext.data import Field


class MatrixDataset(data.Dataset):
    def __init__(self, dataset: data.Dataset, field: Field, **kwargs) -> None:
        fields = [('src', field), ('trg', field),
                  ('diff_alignment', field), ('diff_prev', field), ('diff_updated', field),
                  ('ids', Field(sequential=False, use_vocab=False))]
        examples = []
        for i, src_example in enumerate(dataset.examples):
            for j, diff_example in enumerate(dataset.examples):
                idx = i * len(dataset) + j
                examples.append(data.Example.fromlist(
                    [src_example.src, src_example.trg,
                     diff_example.diff_alignment, diff_example.diff_prev, diff_example.diff_updated,
                     idx], fields))
        super(MatrixDataset, self).__init__(examples, fields, **kwargs)
