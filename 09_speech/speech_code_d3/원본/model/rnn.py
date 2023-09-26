from typing import Tuple
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GRULayer(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.bn = nn.BatchNorm1d(input_dim)
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=1, bias=True,
                          batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x:                   (batch_size, seq_len, input_dim)
        :param lengths:             (batch_size,)                       long
        :return:
                output:             (batch_size, seq_len, hidden_dim)
                state:              (2, batch_size, hidden_dim) GRU states: 2 is for bidirectional
        """
        b, s, d = x.shape
        assert d == self.input_dim

        x = x.view(b * s, d)
        x = self.bn(x)
        x = x.view(b, s, d)

        x = pack_padded_sequence(x, lengths.cpu().int(), batch_first=True, enforce_sorted=False)
        x, state = self.rnn(x)  # (b, s, 2 * h), (2, b, h)
        x, _ = pad_packed_sequence(x, batch_first=True)

        # handle bidirectional
        x = x.view(b, s, 2, -1)  # (b, s, 2, h)
        x = torch.sum(x, dim=2)  # (b, s, h)
        return x, state


class GRUBody(nn.Module):

    def __init__(self, num_layers: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        rnn_layers = []
        rnn_dim = input_dim
        for i in range(num_layers):
            rnn_layers.append(GRULayer(rnn_dim, hidden_dim))
            rnn_dim = hidden_dim

        self.rnn_layers = nn.ModuleList(rnn_layers)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        states = []
        for i in range(self.num_layers):
            x, s = self.rnn_layers[i](x, lengths)
            states.append(s)

        states = torch.cat(states, dim=0)  # (2 * num_layers, b, h)
        return x, states
