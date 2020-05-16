from typing import Tuple, List

import torchtext
from torch import Tensor
from torchtext import data

from neural_editor.seq2seq.config import Config


class ClassifierBatch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(self, original_src: Tuple[Tensor, Tensor], edit_src: Tuple[Tensor, Tensor],
                 trg: Tensor, pad_index: int) -> None:
        original_src, original_src_lengths = original_src
        edit_src, edit_src_lengths = edit_src

        self.original_src = original_src
        self.original_src_lengths = original_src_lengths
        self.original_src_mask = (original_src != pad_index).unsqueeze(-2)
        self.nseqs = original_src.size(0)

        self.edit_src = edit_src  # B * SrcSeqLen
        self.edit_src_lengths = edit_src_lengths  # B
        self.edit_src_mask = (edit_src != pad_index).unsqueeze(-2)

        self.trg = trg

    def __len__(self) -> int:
        return self.nseqs


def rebatch_classifier_iterator(data_iterator: data.Iterator, pad_idx: int) -> List[ClassifierBatch]:
    return [rebatch_classifier(pad_idx, batch) for batch in data_iterator]


def rebatch_classifier(pad_idx: int, batch: torchtext.data.Batch) -> ClassifierBatch:
    """Wrap torchtext batch into our own Batch class for pre-processing"""
    # These fields are added dynamically by PyTorch
    return ClassifierBatch(batch.original_src, batch.edit_src, batch.trg, pad_idx)
