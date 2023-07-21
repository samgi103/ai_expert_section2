from typing import Optional
import os
import torch
from torch.optim.adam import Adam

from wrapper import DeepSpeech2Wrapper
from data.dataset import AudioFolderDataset
from data.featurizer import FilterBankFeaturizer
from data.tokenizer import GraphemeTokenizer
from utils import pad_sequence, calculate_cer, calculate_wer


def build_model(verbose: bool = True):
    # Define your model
    # - reference parameters:
    #       5 rnn layers, 768 hidden dim.


    if verbose:
        count = 0
        s = "Parameters:\n"
        for param_name, param in model.model.named_parameters():
            s += f"... {param_name:<60}\t{tuple(param.shape)}\n"
            count += param.numel()
        s += f"Total parameters: {count}"
        print(s)

    return model


def build_dataset(is_train: bool = False):
    # Featurizer Configure
    #  - audio sample rate: 16kHz
    #  - STFT: 25 ms window and 10 ms stride
    #  - pre-emphasize: 0.95 or 0.97
    #  - number of Mel-filter banks: 80
    #  - use agmentation when training
    #  - number of mask: pick in [1, 2, 3]
    #  - mask size: pick in [4, 6, 8]
    
    
    # Tokenizer Configure:
    # - pad_token="<BLK>"
    
    
    # Dataset: sample audios in 'samples' dir
    
    
    return dataset, tokenizer


def run(is_train: bool = True,
        device="cuda",
        verbose: bool = False,
        checkpoint_path: Optional[str] = None,
        beam_width: int = 20):
    # ---------------------------------------------------------------- #
    # Build
    print("-" * 64)
    # ---------------------------------------------------------------- #
    model = build_model(verbose=verbose)
    model.to(device)
    dataset, tokenizer = build_dataset(is_train)
    # in this example, we need overfitting, so we disable SpecAug.

    # ---------------------------------------------------------------- #
    # Load data and create batch
    print("-" * 64)
    # ---------------------------------------------------------------- #
    # in this example, we simply collect all samples in dataset
    # for actual training, we should use dataloader to random sample mini-batch
    features = []
    labels = []
    scripts = []
    for i in range(len(dataset)):
        data = dataset[i]
        features.append(data[0])
        labels.append(data[1])
        scripts.append(data[2])

    features, feature_lengths = pad_sequence(features)
    labels, label_lengths = pad_sequence(labels)
    if verbose:
        print("Features shape:", features.shape)
        print("Feature lengths:", feature_lengths)
        print("Labels shape:", labels.shape)
        print("Label lengths:", label_lengths)
    batch_size = labels.shape[0]

    features = features.to(device)
    feature_lengths = feature_lengths.to(device)
    labels = labels.to(device)
    label_lengths = label_lengths.to(device)

    # ---------------------------------------------------------------- #
    # Train
    print("-" * 64)
    # ---------------------------------------------------------------- #
    if is_train:
        print("Train start!")
        model.train()
        # in this example, we simply train network to overfit the data
        optimizer = Adam(model.parameters(), lr=0.001)
        for epoch_i in range(100):
            optimizer.zero_grad(set_to_none=True)

            loss = model(features, labels, feature_lengths, label_lengths)
            loss.backward()
            if (epoch_i + 1) % 5 == 0:
                print(f"... [{epoch_i + 1}/100] CTC loss: {loss.item():.6f}")
            optimizer.step()

            # save parameter
            if (epoch_i + 1) % 25 == 0:
                torch.save(model.state_dict(), f"result/checkpoint_{epoch_i + 1}.pth")

        print("Train Done!")
        return  # exit

    else:
        # eval
        print(f"Checkpoint loaded: {checkpoint_path} (beam width: {beam_width})")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # ---------------------------------------------------------------- #
    # Decode
    print("-" * 64)
    # ---------------------------------------------------------------- #
    model.eval()

    # greedy decode
    decoded, logp_scores = model.decode(features, feature_lengths, beam_width=beam_width)
    prediction = [tokenizer.decode(indices) for indices in decoded]

    cer_distance_sum = 0
    cer_length_sum = 0
    wer_distance_sum = 0
    wer_length_sum = 0

    for i in range(batch_size):
        cer, cer_distance, cer_length = calculate_cer(prediction[i], scripts[i])
        wer, wer_distance, wer_length = calculate_wer(prediction[i], scripts[i])
        print(f"... [{i}/{batch_size}] predict: {prediction[i]}\n"
              f"...... target: {scripts[i]}\n"
              f"...... CER: {cer:.3f}, WER: {wer:.3f}")
        cer_distance_sum += cer_distance
        cer_length_sum += cer_length
        wer_distance_sum += wer_distance
        wer_length_sum += wer_length

    # ---------------------------------------------------------------- #
    # Done
    print("-" * 64)
    # ---------------------------------------------------------------- #
    total_cer = cer_distance_sum / cer_length_sum
    total_wer = wer_distance_sum / wer_length_sum
    print(f"Total CER: {total_cer:.3f}, WER: {total_wer:.3f}")


if __name__ == '__main__':
    os.makedirs("result", exist_ok=True)

    # train setting
    run(is_train=True, device="cuda", verbose=True)

    # evaluation with greedy decoding
    run(is_train=False, device="cuda", checkpoint_path="result/checkpoint_25.pth", beam_width=1)  
    run(is_train=False, device="cuda", checkpoint_path="result/checkpoint_50.pth", beam_width=1)  
    run(is_train=False, device="cuda", checkpoint_path="result/checkpoint_75.pth", beam_width=1)  
    run(is_train=False, device="cuda", checkpoint_path="result/checkpoint_100.pth", beam_width=1)  

    # evaluation with beamsearch decoding (improve results only for well-trained cases)
    run(is_train=False, device="cuda", checkpoint_path="result/checkpoint_75.pth", beam_width=20) 
    run(is_train=False, device="cuda", checkpoint_path="result/checkpoint_100.pth", beam_width=20)  
