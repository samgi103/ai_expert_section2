import torch
import torch.nn as nn


class CTCHead(nn.Module):

    def __init__(self, feature_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.vocab_size = vocab_size

        self.bn = nn.BatchNorm1d(feature_dim)
        self.fc = nn.Linear(feature_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x:       (batch_size, seq_len, feature_dim)
        :return:
                output: (batch_size, seq_len, vocab_size)
        """
        b, s, d = x.shape
        assert d == self.feature_dim

        x = x.view(b * s, d)
        x = self.bn(x)  # (b * s, d)
        x = self.fc(x)  # (b * s, d) -> (b * s, v)
        x = x.view(b, s, self.vocab_size)  # (b, s, v)
        return x
