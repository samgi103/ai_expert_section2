from typing import Tuple, Optional
import torch
import torch.nn as nn

from model.subsampling import ConvSubsampling4x2
from model.rnn import GRUBody
from model.head import CTCHead


class DeepSpeech2Model(nn.Module):

    def __init__(self,
                 feature_dim: int,
                 num_rnn_layers: int,
                 hidden_dim: int,
                 vocab_size: int) -> None:
        super().__init__()
        
        self.conv = ConvSubsampling4x2(feature_dim, hidden_dim)
        self.rnn = GRUBody(num_rnn_layers, hidden_dim, hidden_dim)
        self.head = CTCHead(hidden_dim, vocab_size)
        

    def forward(self,
                x: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        - input   x      : (batch_size, seq_length, feature_dim)       float
                  lengths: (batch_size,)                               long
        - output  x      : (batch_size, seq_length // 4, vocab_size)   float
                  lengths: (batch_size,)                               long
        """
        if lengths is None:
            max_seq_length = x.shape[1]
            lengths = torch.full(
                (x.shape[0],), max_seq_length, dtype=torch.long, device=x.device)
            
        # Connect layers to produce proper ouput from input spectrogram
        # input x -> Conv -> GRU -> CTCHead -> head_output
        x, lengths = self.cnn(x, lengths)
        x, _ = self.rnn(x, lengths)
        x = self.head(x)

        return x, lengths





