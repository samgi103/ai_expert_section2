from typing import List
from collections import OrderedDict
import torch

ENGLISH_GRAPHEMES = (
    "_", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
    "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"
)


class GraphemeTokenizer(object):

    def __init__(self, pad_token: str = "<B>", lowercase: bool = False):
        super().__init__()

        vocab = ENGLISH_GRAPHEMES
        if lowercase:
            vocab = tuple(v.lower() for v in vocab)
            pad_token = pad_token.lower()
        else:
            vocab = tuple(v.upper() for v in vocab)
            pad_token = pad_token.upper()

        vocab = (pad_token,) + vocab  # 29
        # pad_token will be used as both "pad" and "blank"
        self.vocab = vocab

        token_to_idx = OrderedDict()
        idx_to_token = OrderedDict()
        for i, v in enumerate(vocab):
            token_to_idx[v] = i
            idx_to_token[i] = v

        self.token_to_idx = token_to_idx
        self.idx_to_token = idx_to_token
        self.pad_token = pad_token
        self.lowercase = lowercase

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, script: str) -> List[int]:
        script = script.strip().replace("\n", "").replace(" ", "_")
        if self.lowercase:
            script = script.lower()
        else:
            script = script.upper()

        chars = [c for c in script]
        indices = [self.token_to_idx[c] for c in chars]
        return indices

    def decode(self, sequence: List[int]) -> str:
        chars = [self.idx_to_token[i] for i in sequence]
        utterance = "".join(chars).replace("_", " ")
        utterance = utterance.replace(self.pad_token, "")
        return utterance

    def __call__(self, script: str) -> torch.Tensor:
        indices = self.encode(script)
        return torch.tensor(indices, dtype=torch.long)


if __name__ == '__main__':
    gp = GraphemeTokenizer()
    print(gp.vocab, len(gp))
    print(gp.token_to_idx)
    print(gp.idx_to_token)