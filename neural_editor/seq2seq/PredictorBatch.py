from typing import Tuple

import torchtext
from torch import Tensor
from torchtext.data import Dataset

from neural_editor.batch_utils import split_sequences_on_hunks
from neural_editor.seq2seq.config import Config


class PredictorBatch:
    def __init__(self, src: Tuple[Tensor, Tensor], trg: Tensor,
                 diff_alignment: Tuple[Tensor, Tensor],
                 diff_prev: Tuple[Tensor, Tensor], diff_updated: Tuple[Tensor, Tensor],
                 updated: Tuple[Tensor, Tensor],
                 ids: Tensor, dataset: Dataset, config: Config) -> None:
        self.hunk_index = dataset.fields['src'].vocab.stoi[config['HUNK_TOKEN']]
        self.pad_index = dataset.fields['src'].vocab.stoi[config['PAD_TOKEN']]
        self.sos_index = dataset.fields['src'].vocab.stoi[config['SOS_TOKEN']]
        self.eos_index = dataset.fields['src'].vocab.stoi[config['EOS_TOKEN']]

        src, src_lengths = src  # B * SrcSeqLen, B

        self.diff_alignment, self.diff_alignment_lengths = diff_alignment  # B * SeqAlignedLen, B
        self.diff_prev, self.diff_prev_lengths = diff_prev  # B * SeqAlignedLen, B
        self.diff_updated, self.diff_updated_lengths = diff_updated  # B * SeqAlignedLen, B
        self.updated, self.updated_lengths = updated

        self.src = src  # B * SrcSeqLen
        self.src_lengths = src_lengths  # B
        self.nseqs = src.size(0)

        self.ids = ids
        self.trg = trg

        self.hunk_numbers = (self.src == self.hunk_index).sum(dim=-1)
        self.diff_alignment_hunks, self.diff_alignment_hunk_lengths = \
            split_sequences_on_hunks(self.diff_alignment, self.hunk_index, self.pad_index,
                                     self.sos_index, self.eos_index)
        self.diff_prev_hunks, self.diff_prev_hunk_lengths = \
            split_sequences_on_hunks(self.diff_prev, self.hunk_index, self.pad_index,
                                     self.sos_index, self.eos_index)
        self.diff_updated_hunks, self.diff_updated_hunk_lengths = \
            split_sequences_on_hunks(self.diff_updated, self.hunk_index, self.pad_index,
                                     self.sos_index, self.eos_index)
        self.src_hunks, self.src_hunk_lengths = \
            split_sequences_on_hunks(self.src, self.hunk_index, self.pad_index,
                                     self.sos_index, self.eos_index)

    def __len__(self) -> int:
        return self.nseqs


def rebatch_predictor(batch: torchtext.data.Batch, dataset: Dataset, config: Config) -> PredictorBatch:
    """Wrap torchtext batch into our own Batch class for pre-processing"""
    # These fields are added dynamically by PyTorch
    return PredictorBatch(batch.src, batch.trg, batch.diff_alignment,
                          batch.diff_prev, batch.diff_updated, batch.updated, batch.ids, dataset, config)
