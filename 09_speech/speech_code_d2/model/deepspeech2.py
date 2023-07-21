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
        # TO DO
        # Define DeepSpeech2 Layers here
        # with 'ConvSubsampling4x2', 'GRUBody', 'CTCHead' module
        # self.conv_subsample = ConvSubsampling4x2(feature_dim, hidden_dim)
        # self.rnn_body = GRUBody(hidden_dim, num_rnn_layers, hidden_dim)
        # self.ctc_head = CTCHead(hidden_dim, vocab_size)
        self.conv = ConvSubsampling4x2(feature_dim, hidden_dim)
        # self.rnn_body = GRUBody(hidden_dim, num_rnn_layers, hidden_dim)
        # self.rnn_body = GRUBody(num_rnn_layers, self.conv, hidden_dim)
        self.rnn_body = GRUBody(num_rnn_layers, hidden_dim, hidden_dim)
        # self.ctc_head = CTCHead(hidden_dim, vocab_size)
        self.ctc_head = CTCHead(hidden_dim , vocab_size)

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

        # Connect layers to produce proper output from input spectrogram
        # input x -> Conv -> GRU -> CTCHead -> head_output
        # TO DO
        # x = self.conv_subsample(x, lengths)
        # x = self.rnn_body(x)
        # head_output = self.ctc_head(x)
        # x = self.conv(x, lengths)
        # x = self.rnn_body(x)        
        # head_output = self.ctc_head(x)
        cnn_output, cnn_output_length = self.conv()
        rnn_output, rnn_state = self.rnn_body()
        head_output = self.head()

        return head_output, lengths





