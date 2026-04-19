import torch
import matplotlib.pyplot as plt
import numpy as np
from next_frame_classifier import NextFrameClassifier, NextFrameClassifier_without_recons
from argparse import Namespace
import dill
import torchaudio
import os
import random
from utils import detect_peaks, max_min_norm
# %%
# =========================
# Load PHN
# =========================
def load_phn(phn_path):
    boundaries = []
    with open(phn_path, "r") as f:
        for line in f:
            start, end, phn = line.strip().split()
            boundaries.append((int(start), int(end), phn))
    return boundaries


# =========================
# Config
# =========================
sr = 16000
hop = 160

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr,
    n_fft=512,
    win_length=480,
    hop_length=hop,
    n_mels=80
)

# =========================
# Checkpoints
# =========================
ckpt_paths = [
    '/home/sunil/prom_based/scpc_ckpts/2026-04-13_10-06-41-unsupseg_test_recons_no_norm/epoch=4.ckpt'
]

ckpt_paths_without_recons = [
    '/home/sunil/prom_based/scpc_ckpts/2026-04-13_10-00-31-unsupseg_test_without_recons_no_norm/epoch=45.ckpt'
]

model_names = ["With Recons", "Without Recons"]

models = []
peak_params = []

# ---- Load WITH recons ----
for ckpt_path in ckpt_paths:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    hp = Namespace(**dict(ckpt["hparams"]))
    model = NextFrameClassifier(hp)

    weights = {k.replace("NFC.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(weights)
    model.eval()

    params = dill.loads(ckpt['peak_detection_params'])['cpc_1']

    models.append(model)
    peak_params.append(params)

# ---- Load WITHOUT recons ----
for ckpt_path in ckpt_paths_without_recons:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    hp = Namespace(**dict(ckpt["hparams"]))
    model = NextFrameClassifier_without_recons(hp)

    weights = {k.replace("NFC.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(weights)
    model.eval()

    params = dill.loads(ckpt['peak_detection_params'])['cpc_1']

    models.append(model)
    peak_params.append(params)

# =========================
# Load random audio
# =========================
folder = '/home/sunil/nani/sup_pred/timit_dp/test'
files = [f for f in os.listdir(folder) if f.endswith('wav')]
audio_path = os.path.join(folder, random.choice(files))

print(f"Chosen file: {audio_path}")

phn_path = audio_path.replace(".wav", ".phn")
phn_data = load_phn(phn_path)

audio, sr = torchaudio.load(audio_path)
audio = audio[0]

# =========================
# Ground truth boundaries
# =========================
gt_boundaries = [end / sr for _, end, _ in phn_data]

# =========================
# Compute log-mel
# =========================
mel = mel_transform(audio)
log_mel = torch.log(mel + 1e-8).transpose(0, 1)

# =========================
# Time axes
# =========================
audio_np = audio.numpy()
time_wave = np.linspace(0, len(audio_np) / sr, len(audio_np))

log_mel_np = log_mel.numpy().T

# =========================
# Plot BOTH models (no blocking)
# =========================
figs = []

for i, model in enumerate(models):

    # =========================
    # Forward pass
    # =========================
    if i == len(models) - 1:
        preds, _ = model(log_mel.unsqueeze(0), [log_mel.size(0)])
    else:
        preds, _, _ = model(log_mel.unsqueeze(0), [log_mel.size(0)])

    preds = preds[1][0]

    scores = 1 - max_min_norm(preds)

    peaks = detect_peaks(
        x=scores,
        lengths=[scores.shape[1]],
        prominence=peak_params[i]["prominence"],
        width=peak_params[i]["width"],
        distance=peak_params[i]["distance"]
    )[0]

    peaks = peaks * hop / sr

    # =========================
    # Plot
    # =========================
    fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # ---- Waveform ----
    axs[0].plot(time_wave, audio_np, linewidth=0.8)
    axs[0].set_title(f"{model_names[i]} - Waveform")

    # GT boundaries
    for t in gt_boundaries:
        axs[0].axvline(t, color='black', linewidth=2)

    # Predicted peaks
    for t in peaks:
        axs[0].axvline(t, color='red', linestyle='--', alpha=0.7)

    # Centered phoneme labels
    y_top = axs[0].get_ylim()[1]
    for start, end, phn in phn_data:
        mid = (start + end) / 2 / sr
        axs[0].text(mid, y_top * 0.9,
                    phn, ha='center', va='top', fontsize=9)

    # ---- Log-mel ----
    im = axs[1].imshow(
        log_mel_np,
        aspect='auto',
        origin='lower',
        extent=[0, time_wave[-1], 0, log_mel_np.shape[0]]
    )

    axs[1].set_title(f"{model_names[i]} - Log-Mel")

    # GT boundaries
    for t in gt_boundaries:
        axs[1].axvline(t, color='black', linewidth=2)

    # Predicted peaks
    for t in peaks:
        axs[1].axvline(t, color='red', linestyle='--', alpha=0.7)

    # Centered phoneme labels
    for start, end, phn in phn_data:
        mid = (start + end) / 2 / sr
        axs[1].text(mid,
                    log_mel_np.shape[0] + 2,
                    phn,
                    ha='center', va='bottom', fontsize=9)

    # Align axes perfectly
    axs[0].set_xlim(0, time_wave[-1])
    axs[1].set_xlim(0, time_wave[-1])

    plt.colorbar(im, ax=axs[1])

    figs.append(fig)

# =========================
# Show BOTH plots together
# =========================
plt.show()

# %%