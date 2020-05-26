from typing import Tuple

import torchtext
from torch import Tensor


class PredictorBatch:
    def __init__(self, src: Tuple[Tensor, Tensor], trg: Tensor,
                 diff_alignment: Tuple[Tensor, Tensor],
                 diff_prev: Tuple[Tensor, Tensor], diff_updated: Tuple[Tensor, Tensor],
                 updated: Tuple[Tensor, Tensor],
                 ids: Tensor) -> None:
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

    def __len__(self) -> int:
        return self.nseqs


def rebatch_predictor(batch: torchtext.data.Batch) -> PredictorBatch:
    """Wrap torchtext batch into our own Batch class for pre-processing"""
    # These fields are added dynamically by PyTorch
    return PredictorBatch(batch.src, batch.trg, batch.diff_alignment,
                          batch.diff_prev, batch.diff_updated, batch.updated, batch.ids)
