import torch
from torch import nn, Tensor
from torch.nn import Embedding

from edit_representation.sequence_encoding import EditEncoder


class GoodEditClassifier(nn.Module):
    def __init__(self, original_src_encoder: EditEncoder, edit_src_encoder: EditEncoder,
                 embedding: Embedding, output_size: int) -> None:
        super(GoodEditClassifier, self).__init__()
        self.original_src_encoder = original_src_encoder
        self.edit_src_encoder = edit_src_encoder
        self.embedding = embedding
        self.output_size = output_size
        self.bilinear_layer = nn.Bilinear(original_src_encoder.hidden_size * 2, edit_src_encoder.hidden_size * 2,
                                          output_size)

    def forward(self, batch) -> Tensor:
        _, (original_src_hidden, _) = self.original_src_encoder(self.embedding(batch.original_src),
                                                                batch.original_src_mask, batch.original_src_lengths)
        original_src_hidden = original_src_hidden[-1]
        _, (edit_src_hidden, _) = self.edit_src_encoder(self.embedding(batch.edit_src),
                                                        batch.edit_src_mask, batch.edit_src_lengths)
        edit_src_hidden = edit_src_hidden[-1]
        out = self.bilinear_layer(original_src_hidden, edit_src_hidden)
        return out.squeeze(dim=1)

    def predict(self, batch) -> Tensor:
        return torch.sigmoid(self.forward(batch))
