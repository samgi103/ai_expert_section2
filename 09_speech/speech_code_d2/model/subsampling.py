from typing import Optional, Tuple
import torch
import torch.nn as nn


@torch.no_grad()
def make_mask_by_length(lengths: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    assert lengths.ndim == 1
    batch_size = lengths.shape[0]

    if max_length is None:
        max_length = lengths.max().item()

    device = lengths.device

    seq_range = torch.arange(0, max_length, dtype=torch.long, device=device)  # (s,)
    seq_range = seq_range.unsqueeze(0).expand(batch_size, max_length)  # (b, s)
    seq_length = lengths.unsqueeze(1).expand(batch_size, max_length)  # (b, s)
    mask = torch.less(seq_range, seq_length)  # (b, s)
    return mask


@torch.no_grad()
def fix_lengths_after_conv(lengths: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    new_lengths = torch.floor((lengths + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
    return new_lengths.long()


class ConvSubsampling4x2(nn.Module):

    def __init__(self,
                 feature_dim: int,
                 out_dim: Optional[int]) -> None:
        super().__init__()
        assert feature_dim % 4 == 0
        self.feature_dim = feature_dim

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.Hardtanh(0, 20, inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5), bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = nn.Hardtanh(0, 20, inplace=True)

        # if feature dimension is 80, it is converted as:
        # (channel, dim)
        # conv1: (1, 80) -> (32, 40)
        # conv2: (32, 40) -> (32, 40)
        out_feature_dim = 32 * feature_dim // 2

        # this linear is not in DS2 paper, but it is common practice in recent works.
        if out_dim is not None:
            self.out_dim = out_dim
            self.linear = nn.Linear(out_feature_dim, out_dim, bias=False)  # will be followed by BN
        else:
            self.out_dim = out_feature_dim
            self.linear = None

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x:               (batch_size, seq_length, feature_dim)       float
        :param lengths:         (batch_size,)                               long
        :return:
                output:         (batch_size, seq_length // 4, out_dim)
                out_lengths:    (batch_size,)                               long
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)

        batch_size = x.shape[0]

        # we keep mask out invalid parts to ensure that the result does not depends on the sequence length.

        # first convolution input should be already zero-ed for invalid regions.
        x = self.conv1(x)  # (b, 1, s, d) -> (b, 32, s//2, d//2)
        lengths = fix_lengths_after_conv(lengths, 41, 2, 20)
        mask = make_mask_by_length(lengths)  # (b, s//2)
        x = x * mask.view(batch_size, 1, -1, 1)

        x = self.bn1(x)
        x = x * mask.view(batch_size, 1, -1, 1)  # bn does not change mask length
        x = self.act1(x)  # HardTanh keep 0 as 0

        x = self.conv2(x)  # (b, 32, s//2, d//2) -> (b, 32, s//4, d//2)
        lengths = fix_lengths_after_conv(lengths, 21, 2, 10)
        mask = make_mask_by_length(lengths)  # (b, s//4)
        x = x * mask.view(batch_size, 1, -1, 1)

        x = self.bn2(x)
        x = x * mask.view(batch_size, 1, -1, 1)  # bn does not change mask length
        x = self.act2(x)  # HardTanh keep 0 as 0

        _, c, t, f = x.shape
        # aggregate feature for each time-step
        x = x.transpose(1, 2).contiguous().view(batch_size, t, c * f)
        if self.linear is not None:
            x = self.linear(x)  # (b, s//4, 32 * d//2) -> (b, s//4, out_dim)
            x = x * mask.view(batch_size, -1, 1)
        return x, lengths
