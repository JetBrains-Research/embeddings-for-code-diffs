from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from neural_editor.seq2seq import Batch


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.vocab_projection = nn.Linear(hidden_size, vocab_size, bias=False)  # DecoderH -> V

    def forward(self, x: Tuple[Tensor, Tensor, Tensor], batch: Batch) -> Tensor:
        """
        Projects hidden representation to vocabulary size vector and then softmax to probabilities.
        :param x: [B, TrgSeqLen, DecoderH]
        :return: [B, TrgSeqLen, V]
        """
        p_gen = x[1].unsqueeze(-1)
        # TODO: find the way to use log_softmax
        vocab_dist = p_gen * F.softmax(self.vocab_projection(x[0]), dim=-1)
        copy_dist = (1 - p_gen) * x[2]

        final_dist = torch.cat((vocab_dist,
                                torch.zeros(vocab_dist.shape[0], vocab_dist.shape[1], batch.oov_num).to(x[0].device)),
                               dim=-1)
        final_dist.scatter_add_(dim=-1,
                                index=batch.scatter_indices.unsqueeze(1).repeat((1, copy_dist.shape[1], 1)).long(),
                                src=copy_dist)
        # get rid of zero probabilities
        # TODO: think about this problem: we have two src: src1 and src2. src2 contains oov word 'synchronized',
        # and src1 doesn't. But trg1 contains 'sync' and
        # when we compute loss we have probability zero for 'synchronized'
        # because src1 didn't contain this word. Should we set <unk> as target token or just add eps for probability?
        final_dist[final_dist == 0] += 1e-20
        return torch.log(final_dist)
