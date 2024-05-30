import os
import torch
import utils
from text import text_to_sequence
from models import SynthesizerTrn
from scipy.io.wavfile import write
import numpy as np
from text.symbols import symbols


def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(
        checkpoint_path), f"Checkpoint file not found: {checkpoint_path}"
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['model'])
    if optimizer:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    return checkpoint_dict


def infer(text, model, hps, device):
    model.eval()
    sequence = text_to_sequence(text, hps.data.text_cleaners)
    sequence = torch.LongTensor(sequence).unsqueeze(0).to(device)
    with torch.no_grad():
        x_lengths = torch.LongTensor([sequence.size(1)]).to(device)
        y_hat, *_ = model.infer(sequence, x_lengths)
        y_hat = y_hat.squeeze().cpu().numpy()
        return y_hat


def save_wav(y_hat, filename, sampling_rate):
    y_hat = (y_hat * 32767).astype(np.int16)
    write(filename, sampling_rate, y_hat)


if __name__ == "__main__":
    hps = utils.get_hparams_from_file('configs/ljs_base.json')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = SynthesizerTrn(len(symbols), hps.data.filter_length // 2 + 1,
                           hps.train.segment_size // hps.data.hop_length, **hps.model).to(device)

    # Path to your trained checkpoint
    # Update this to the actual checkpoint path
    checkpoint_path = './checkpoints/G_latest.pth'
    print(f"Loading checkpoint from {checkpoint_path}")
    load_checkpoint(checkpoint_path, model)
    print("Model loaded from checkpoint")

    # Text input
    text = "你好，欢迎使用这个语音合成模型。"  # Your Chinese text here

    # Generate speech
    y_hat = infer(text, model, hps, device)
    print(f"Generated output shape: {y_hat.shape}")
    print(
        f"Generated output max value: {y_hat.max()}, min value: {y_hat.min()}")

    # Check if y_hat is silent (all zeros)
    if np.all(y_hat == 0):
        print("Warning: Generated output is silent (all zeros).")

    # Save the output
    output_wav_path = 'output.wav'
    save_wav(y_hat, output_wav_path, hps.data.sampling_rate)
    print(f"Generated speech saved to {output_wav_path}")
