# %%
%matplotlib inline
import torch
import torchaudio
import random
import dill
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import Namespace
import argparse

# ---- your imports ----
from next_frame_classifier import NextFrameClassifier
from utils import detect_peaks, replicate_first_k_frames, max_min_norm

# ---- MEL TRANSFORM ----
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=512,
    win_length=480,
    hop_length=160,
    n_mels=80
)

hop_length = 160


# ---- LOAD PHN (TIMIT style) ----
def load_phn(phn_path):
    segments = []
    with open(phn_path, "r") as f:
        for line in f:
            start, end, phoneme = line.strip().split()
            segments.append((int(start), int(end), phoneme))
    return segments


# ---- MAIN ----
def main(wav_dir, ckpt_path, prominence=None):

    # ---- PICK RANDOM FILE ----
    dir_path = Path(wav_dir)
    files = [f for f in dir_path.iterdir() if f.suffix == ".wav"]
    wav = random.choice(files)

    print(f"Running inference on: {wav.name}")
    print(f"Using checkpoint: {ckpt_path} that has reconstruction")
    print("\n" + "-" * 90)

    # ---- LOAD CHECKPOINT ----
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hp = Namespace(**dict(ckpt["hparams"]))

    model = NextFrameClassifier(hp)
    weights = {k.replace("NFC.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(weights)
    model.eval()

    peak_detection_params = dill.loads(ckpt['peak_detection_params'])['cpc_1']

    if prominence is not None:
        print(f"Overriding prominence with {prominence}")
        peak_detection_params["prominence"] = prominence

    print("Prominence:", peak_detection_params["prominence"])

    # ---- LOAD AUDIO ----
    audio, sr = torchaudio.load(wav)
    assert sr == 16000, "Audio must be 16kHz"
    audio = audio[0]  # (T,)

    # ---- MEL + LOG MEL ----
    mel_spect = mel_transform(audio)              # (n_mels, T)
    log_mel = torch.log(mel_spect + 1e-8)
    log_mel_T = log_mel.transpose(0, 1).cpu()     # (T, n_mels)

    # ---- MODEL INFERENCE ----
    with torch.no_grad():
        proxy_mel, frame_level_rep, seg_rep, durations, latent_vec, pred_boundaries, mask, d ,V, W_int, preds = model(log_mel_T.unsqueeze(0), torch.tensor([log_mel_T.shape[0]]))

    # ---- PROCESS PREDICTIONS ----
    preds = preds[1][0]
    preds = replicate_first_k_frames(preds, k=1, dim=1)
    preds = 1 - max_min_norm(preds)

    peak_indices = detect_peaks(
        x=preds,
        lengths=[preds.shape[1]],
        prominence=peak_detection_params["prominence"],
        width=peak_detection_params["width"],
        distance=peak_detection_params["distance"]
    )[0]

    # ---- PROCESS PROXY MEL ----
    proxy_mel = proxy_mel.squeeze(0).cpu()  # (T, n_mels)

    # ---- LOAD GROUND TRUTH ----
    phn_path = wav.with_suffix(".phn")
    if phn_path.exists():
        gt_segments = load_phn(phn_path)
    else:
        print("No .phn file found!")
        gt_segments = []

    # ---- PLOTTING ----
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # =========================================================
    # ===== ORIGINAL LOG MEL + GT ==============================
    # =========================================================
    axs[0].imshow(log_mel_T.T, aspect='auto', origin='lower')
    axs[0].set_title("Original Log-Mel + Ground Truth")

    for (start, end, phoneme) in gt_segments:

        # convert to frame indices
        start_f = start / hop_length
        end_f = end / hop_length
        center_f = (start_f + end_f) / 2

        # draw boundary
        axs[0].axvline(x=end_f, color='white', linestyle='--', linewidth=1)

        # draw phoneme label (centered)
        if end_f - start_f > 2:  # avoid clutter for tiny segments
            axs[0].text(
                center_f,
                log_mel_T.shape[1] - 5,
                phoneme,
                color='white',
                fontsize=8,
                ha='center',
                va='top',
                bbox=dict(facecolor='black', alpha=0.5, pad=1)
            )

    # =========================================================
    # ===== PROXY MEL + PREDICTIONS ============================
    # =========================================================
    print(proxy_mel.shape)
    print(proxy_mel.min(), proxy_mel.max())
    print(proxy_mel.mean(), proxy_mel.std())
    axs[1].imshow(proxy_mel.T, aspect='auto', origin='lower', vmin=-1, vmax=1)
    axs[1].set_title("Reconstructed (Proxy Mel) + Predicted Boundaries")

    for idx in peak_indices:
        axs[1].axvline(x=idx, color='red', linestyle='--', linewidth=1)

    axs[1].set_xlabel("Time (frames)")
    # ---- FINALIZE ----
    plt.tight_layout()
    plt.show()
    # plt.close()
# %%


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference and plot mel + boundaries"
    )

    parser.add_argument(
        "--wav_dir",
        type=str,
        required=True,
        help="Path to directory containing wav files"
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )

    parser.add_argument(
        "--prominence",
        type=float,
        default=None,
        help="Override peak detection prominence"
    )

    args = parser.parse_args()

# %%
main(
    wav_dir='../../../ph_sg/timit_dp/test',
    ckpt_path='/home/sunil/prom_based/scpc_ckpts/2026-04-18_22-51-46-gru_scpc_full_timit/epoch=14.ckpt',
    prominence=None
)
# %%
