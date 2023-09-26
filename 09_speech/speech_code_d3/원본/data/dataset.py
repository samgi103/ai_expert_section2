from typing import Tuple
import os
import torch
from torch.utils.data.dataset import Dataset

from data.featurizer import FilterBankFeaturizer
from data.tokenizer import GraphemeTokenizer


class AudioFolderDataset(Dataset):

    def __init__(self,
                 folder_path: str,
                 featurizer: FilterBankFeaturizer,
                 tokenizer: GraphemeTokenizer,
                 audio_ext: str = "flac",
                 text_ext: str = "txt",
                 ):
        super().__init__()
        self.folder_path = folder_path

        data = []
        for f in os.listdir(folder_path):
            audio_p = os.path.join(folder_path, f)
            if not os.path.isfile(audio_p):
                continue

            if audio_p.endswith(audio_ext):
                # audio exist, now check if script also exist
                text_p = audio_p.replace(audio_ext, text_ext)
                if not os.path.isfile(text_p):
                    raise ValueError(f"Script file not exist: {text_p}")
                data.append((audio_p, text_p))

        self.data = data
        self.featurizer = featurizer
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        audio_p, text_p = self.data[idx]

        with open(text_p, "r") as f:
            lines = f.readlines()
            script = lines[0].strip().replace("\n", "")

        feature = self.featurizer(audio_p)
        label = self.tokenizer(script)
        # print(feature.shape, label.shape, script)
        return feature, label, script
