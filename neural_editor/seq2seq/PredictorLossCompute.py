from typing import Optional

from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer


class PredictorLossCompute:
    def __init__(self, criterion: _Loss, optimizer: Optional[Optimizer]) -> None:
        self.criterion = criterion
        self.optimizer = optimizer

    def __call__(self, x: Tensor, y: Tensor) -> float:
        """
        :param x: [B, 1]
        :param y: [B, 1]
        :return: float
        """
        loss = self.get_loss(x, y)

        if self.optimizer is not None:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.data.item()

    def get_loss(self, x, y) -> Tensor:
        loss = self.criterion(x, y.float())
        return loss
