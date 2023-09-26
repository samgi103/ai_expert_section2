from typing import Tuple, List
import torch
import editdistance


def pad_sequence(sequences: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad list of sequence to maximum length.
    :param sequences:               List of (seq_length, ...)
    :return:
            padded sequence:        (batch_size, max_seq_length, ...)
            lengths:                (batch_size,)
    """
    batch_size = len(sequences)
    lengths = [s.shape[0] for s in sequences]
    max_length = max(lengths)
    trailing_shape = sequences[0].shape[1:]

    out_shape = (batch_size, max_length) + trailing_shape
    out_tensor = torch.zeros(out_shape, dtype=sequences[0].dtype, device=sequences[0].device)
    for i, tensor in enumerate(sequences):
        length = tensor.shape[0]
        out_tensor[i, :length, ...] = tensor

    lengths = torch.tensor(lengths, dtype=torch.long, device=out_tensor.device)
    return out_tensor, lengths


def calculate_cer(prediction: str, target: str) -> Tuple[float, int, int]:
    prediction = prediction.replace(" ", "").strip()
    target = target.replace(" ", "").strip()

    distance = editdistance.eval(prediction, target)
    length = len(target)
    return distance / length, distance, length


def calculate_wer(prediction: str, target: str) -> Tuple[float, int, int]:
    prediction = prediction.strip().split(" ")
    target = target.strip().split(" ")

    distance = editdistance.eval(prediction, target)
    length = len(target)
    return distance / length, distance, length
