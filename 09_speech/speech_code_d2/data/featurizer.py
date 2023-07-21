import math
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as AT


class FilterBankFeaturizer(object):

    def __init__(self,
                 sample_rate: int,
                 window_sec: float,
                 stride_sec: float,
                 pre_emphasize: float = 0.97,
                 n_mels: int = 80,
                 use_aug: bool = False,
                 time_mask_num: int = 2,
                 time_mask_size: int = 16,
                 freq_mask_num: int = 1,
                 freq_mask_size: int = 27):
        super().__init__()
        self.sample_rate = sample_rate

        win_length = int(sample_rate * window_sec)
        hop_length = int(sample_rate * stride_sec)
        n_fft = 2 ** math.ceil(math.log2(win_length))

        self.win_length = win_length
        self.hop_length = hop_length
        self.pre_emphasize = pre_emphasize
        self.n_mels = n_mels

        self.mel_spec = AT.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft,
                                          win_length=win_length, hop_length=hop_length, n_mels=n_mels)
        self.amp_to_db = AT.AmplitudeToDB()

        if use_aug:
            spec_aug = []
            for i in range(time_mask_num):
                spec_aug.append(AT.TimeMasking(time_mask_size))
            for i in range(freq_mask_num):
                spec_aug.append(AT.FrequencyMasking(freq_mask_size))
            self.spec_aug = nn.Sequential(*spec_aug)
        else:
            self.spec_aug = None

    @torch.no_grad()
    def __call__(self, audio_path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(audio_path)
        assert sample_rate == self.sample_rate

        # pre-emphasize
        waveform = waveform.float()
        waveform = waveform[..., 1:] - self.pre_emphasize * waveform[..., :-1]

        # log-mel-spectrogram
        feature = self.mel_spec(waveform)  # (1, wave_length) -> (1, num_mels, num_windows)
        feature = self.amp_to_db(feature)

        # normalize feature per sample
        mean = torch.mean(feature, dim=2, keepdim=True)
        std = torch.std(feature, dim=2, keepdim=True, unbiased=False)

        feature = (feature - mean) / (std + 1e-6)

        if self.spec_aug is not None:
            feature = self.spec_aug(feature)

        feature = feature.transpose(1, 2).squeeze(0)  # (num_windows, num_mels)
        assert feature.ndim == 2

        return feature
