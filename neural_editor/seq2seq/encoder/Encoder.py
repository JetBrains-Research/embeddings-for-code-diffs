from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from neural_editor.seq2seq import Batch


class Encoder(nn.Module):
    """Encodes a sequence of word embeddings"""

    def __init__(self, embed: nn.Embedding, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.embed = embed
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, batch: Batch) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Applies a bidirectional LSTM to sequence of embeddings x.
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        :param x: [B, SrcSeqLen, EmbCode]
        :param mask: [B, 1, SrcSeqLen]
        :param lengths: [B]
        :returns: Tuple[
            [B, SrcSeqLen, NumDirections * SrcEncoderH],
            Tuple[[NumLayers, B, NumDirections * SrcEncoderH], [NumLayers, B, NumDirections * SrcEncoderH]]
        ]
        """
        x, mask, lengths = self.embed(batch.src), batch.src_mask, batch.src_lengths
        packed = pack_padded_sequence(x, lengths, batch_first=True)  # [RealTokenNumberWithoutPad, SecSeqLen]
        # packed seq, [NumLayers * NumDirections, B, SrcEncoderH], [NumLayers * NumDirections, B, SrcEncoderH]
        output, (h_n, c_n) = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)  # [B, SrcSeqLen, NumDirections * SrcEncoderH]

        # we need to manually concatenate the final states for both directions
        fwd_final = h_n[0:h_n.size(0):2]  # [NumLayers, B, SrcEncoderH]
        bwd_final = h_n[1:h_n.size(0):2]  # [NumLayers, B, SrcEncoderH]
        h_n = torch.cat([fwd_final, bwd_final], dim=2)  # [NumLayers, B, NumDirections * SrcEncoderH]
        fwd_cell = c_n[0:c_n.size(0):2]  # [NumLayers, B, SrcEncoderH]
        bwd_cell = c_n[1:c_n.size(0):2]  # [NumLayers, B, SrcEncoderH]
        c_n = torch.cat([fwd_cell, bwd_cell], dim=2)  # [NumLayers, B, NumDirections * SrcEncoderH]

        return output, (h_n, c_n)
