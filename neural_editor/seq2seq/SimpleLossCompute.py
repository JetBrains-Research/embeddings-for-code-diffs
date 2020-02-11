from typing import Tuple

from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from neural_editor.seq2seq import Generator, Batch


class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator: Generator, criterion: _Loss, optimizer: Optimizer) -> None:
        self.generator = generator
        self.criterion = criterion
        self.optimizer = optimizer

    def __call__(self, x: Tuple[Tensor, Tensor, Tensor], batch: Batch, norm: int) -> float:
        """
        :param x: [B, TrgSeqLen, DecoderH]
        :param y: [B, TrgSeqLen]
        :param norm: normalizing coefficient (usually batch size)
        :return: float
        """
        x = self.generator(x, batch)  # [B, TrgSeqLen, V]
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              batch.trg_y_extended_vocab.contiguous().view(-1))  # [1]
        loss = loss / norm

        if self.optimizer is not None:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.data.item() * norm
