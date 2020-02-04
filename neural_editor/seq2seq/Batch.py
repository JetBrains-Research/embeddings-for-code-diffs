from typing import Tuple

from torch import Tensor

from neural_editor.seq2seq.config import Config


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(self, src: Tuple[Tensor, Tensor], trg: Tuple[Tensor, Tensor],
                 diff_alignment: Tuple[Tensor, Tensor],
                 diff_prev: Tuple[Tensor, Tensor], diff_updated: Tuple[Tensor, Tensor],
                 pad_index: int, config: Config) -> None:
        src, src_lengths = src  # B * SrcSeqLen, B
        # TODO: remove first sos token
        # src = src[1:]
        # src_lengths = src_lengths - 1

        self.diff_alignment, self.diff_alignment_lengths = diff_alignment  # B * SeqAlignedLen, B
        self.diff_alignment_mask = (self.diff_alignment != pad_index).unsqueeze(-2)  # B * 1 * SeqAlignedLen
        self.diff_prev, self.diff_prev_lengths = diff_prev  # B * SeqAlignedLen, B
        self.diff_prev_mask = (self.diff_prev != pad_index).unsqueeze(-2)  # B * 1 * SeqAlignedLen
        self.diff_updated, self.diff_updated_lengths = diff_updated  # B * SeqAlignedLen, B
        self.diff_updated_mask = (self.diff_updated != pad_index).unsqueeze(-2)  # B * 1 * SeqAlignedLen

        self.src = src  # B * SrcSeqLen
        self.src_lengths = src_lengths  # B
        self.src_mask = (src != pad_index).unsqueeze(-2)  # B * 1 * SrcSeqLen
        self.nseqs = src.size(0)

        self.trg = None
        self.trg_y = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        if trg is not None:
            trg, trg_lengths = trg
            # TODO: what is that? it is padding, why it isn't eos token?
            # Answer: it is problem because not all samples in batch have the same size therefore padding is cut
            self.trg = trg[:, :-1]  # B * (TrgSeqLen - 1), removing eos from sequences
            self.trg_lengths = trg_lengths  # B
            self.trg_y = trg[:, 1:]  # B * (TrgSeqLen - 1), removing sos from sequences
            self.trg_mask = (self.trg_y != pad_index)  # B * (TrgSeqLen - 1)
            self.ntokens = (self.trg_y != pad_index).data.sum().item()

        self.src = self.src.to(config['DEVICE'])
        self.src_mask = self.src_mask.to(config['DEVICE'])

        if trg is not None:
            self.trg = self.trg.to(config['DEVICE'])
            self.trg_y = self.trg_y.to(config['DEVICE'])
            self.trg_mask = self.trg_mask.to(config['DEVICE'])

    def __len__(self) -> int:
        return self.nseqs
