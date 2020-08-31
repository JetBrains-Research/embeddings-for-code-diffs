from typing import Tuple

import torch
from torch import Tensor
from torch import nn

from neural_editor.seq2seq import Batch
from neural_editor.seq2seq.encoder import Encoder


class EncoderPredictor(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """

    def __init__(self, encoder: Encoder) -> None:
        super(EncoderPredictor, self).__init__()
        self.encoder = encoder
        predictor_input_size = encoder.get_hidden_size()
        predictor_hidden_size = int(predictor_input_size / 2)
        self.predictor_net = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(predictor_input_size, predictor_hidden_size),
            nn.ReLU(),
            nn.Linear(predictor_hidden_size, 1)
        )

    def forward(self, batch: Batch) -> Tensor:
        (edit_hidden, _), _, (encoder_hidden, _) = self.encoder(batch)
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


