from torch import nn, Tensor
import torch.nn.functional as F


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super(Generator, self).__init__()
        self.projection = nn.Linear(hidden_size, vocab_size, bias=False)  # DecoderH -> V

    def forward(self, x: Tensor) -> Tensor:
        """
        Projects hidden representation to vocabulary size vector and then softmax to probabilities.
        :param x: [B, TrgSeqLen, DecoderH]
        :return: [B, TrgSeqLen, V]
        """
        return F.log_softmax(self.projection(x), dim=-1)
