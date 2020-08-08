from typing import Tuple

from torch import nn, Tensor

from edit_representation.sequence_encoding import EditEncoder
from neural_editor.seq2seq import Batch
from neural_editor.seq2seq.encoder import SrcEncoder


class Encoder(nn.Module):

    def __init__(self, src_encoder: SrcEncoder, edit_encoder: EditEncoder) -> None:
        super(Encoder, self).__init__()
        self.src_encoder = src_encoder
        self.edit_encoder = edit_encoder
        self.edit_final = None

    def forward(self, batch: Batch) -> Tuple[Tuple[Tensor, Tensor], Tensor, Tuple[Tensor, Tensor]]:
        edit_final = self.edit_encoder.encode_edit(batch)
        src_output, src_final = self.src_encoder(batch)
        return edit_final, src_output, src_final

    def get_hidden_size(self) -> int:
        return self.src_encoder.hidden_size * 2 + self.edit_encoder.hidden_size * 2
