from typing import Optional

from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from neural_editor.seq2seq import Generator


class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator: Generator, criterion: _Loss, optimizer: Optimizer) -> None:
        self.generator = generator
        self.criterion = criterion
        self.optimizer = optimizer

    def __call__(self, x: Tensor, y: Tensor, norm: int, x2: Optional[Tensor] = None) -> float:
        """
        :param x: [B, TrgSeqLen, DecoderH]
        :param y: [B, TrgSeqLen]
        :param norm: normalizing coefficient (usually batch size)
        :return: float
        """
        loss = self.get_loss(x, y)
        if x2 is not None:
            loss += self.get_loss(x2, y)
        loss = loss / norm

        if self.optimizer is not None:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.data.item() * norm

    def get_loss(self, x, y) -> Tensor:
        x = self.generator(x)  # [B, TrgSeqLen, V]
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))  # [1]
        return loss
