from typing import Tuple, Optional
import torch
import torch.nn as nn

from model.subsampling import ConvSubsampling4x2
from model.rnn import GRUBody
from model.head import CTCHead


class DeepSpeech2Model(nn.Module):

    def __init__(self,
                 feature_dim: int,
                 num_layers: int,
                 hidden_dim: int,
                 vocab_size: int) -> None:
        super().__init__()
        # Define DeepSpeech2 Layers here
        # with 'ConvSubsampling4x2', 'GRUBody', 'CTCHead' module

    def forward(self,
                x: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x:               (batch_size, seq_length, feature_dim)       float
        :param lengths:         (batch_size,)                               long
        :return:
                output:         (batch_size, seq_length // 4, vocab_size)
                out_lengths:    (batch_size,)                               long
        """
        if lengths is None:
            max_seq_length = x.shape[1]
            lengths = torch.full((x.shape[0],), max_seq_length, dtype=torch.long, device=x.device)
        # Connect layers to produce proper output from input spectrogram
        # input x -> Conv -> GRU -> CTCHead -> head_output
        

        return x, lengths
