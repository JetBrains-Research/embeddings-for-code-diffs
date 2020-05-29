from typing import Tuple

import torch
from torch import Tensor
from torch import nn

from edit_representation.sequence_encoding import EditEncoder
from neural_editor.seq2seq import Batch
from neural_editor.seq2seq.encoder import Encoder


class EncoderPredictor(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """

    def __init__(self, encoder: Encoder, edit_encoder: EditEncoder) -> None:
        super(EncoderPredictor, self).__init__()
        self.encoder = encoder
        self.edit_encoder = edit_encoder
        predictor_input_size = encoder.hidden_size * 2 + edit_encoder.hidden_size * 2
        predictor_hidden_size = int(predictor_input_size / 2)
        self.predictor_net = nn.Sequential(
            nn.Linear(predictor_input_size, predictor_hidden_size),
            nn.ReLU(),
            nn.Linear(predictor_hidden_size, 1)
        )

    def forward(self, batch: Batch) -> Tensor:
        (edit_hidden, _), _, (encoder_hidden, _) = self.encode(batch)
        encoder_hidden = encoder_hidden[-1]
        edit_hidden = edit_hidden[-1]
        predictor_input = torch.cat((encoder_hidden, edit_hidden), dim=-1)
        out = self.predictor_net(predictor_input)
        return out.squeeze(dim=1)

    def predict(self, batch, labels=False) -> Tensor:
        probs = torch.sigmoid(self.forward(batch))
        return probs if not labels else self.predict_labels(probs)

    def predict_from_forward(self, out: Tensor, labels=False) -> Tensor:
        probs = torch.sigmoid(out)
        return probs if not labels else self.predict_labels(probs)

    def predict_labels(self, probs: Tensor) -> Tensor:
        return torch.round(probs).long()

    def encode_edit(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        """
        Returns edit representations (edit_final) of samples in the batch.
        :param batch: batch to encode
        :return: Tuple[[NumLayers, B, NumDirections * DiffEncoderH], [NumLayers, B, NumDirections * DiffEncoderH]]
        """
        return self.edit_encoder.encode_edit(batch)

    def encode(self, batch: Batch) -> Tuple[Tuple[Tensor, Tensor], Tensor, Tuple[Tensor, Tensor]]:
        """
        Encodes edits and prev sequences
        :param batch: batch to process
        :return: Tuple[
            Tuple[[NumLayers, B, NumDirections * DiffEncoderH], [NumLayers, B, NumDirections * DiffEncoderH]],
            [B, SrcSeqLen, NumDirections * SrcEncoderH],
            Tuple[[NumLayers, B, NumDirections * SrcEncoderH], [NumLayers, B, NumDirections * SrcEncoderH]]
        ]
        """
        edit_final = self.encode_edit(batch)
        encoder_output, encoder_final = self.encoder(batch)
        return edit_final, encoder_output, encoder_final

