import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
torch.multiprocessing.set_sharing_strategy('file_system')
from tqdm import tqdm
import numpy as np
import os
from os.path import join, basename
from boltons.fileutils import iter_find_files
import soundfile as sf
import librosa
import pickle
from multiprocessing import Pool
import random
import torchaudio
import math
from torchaudio.datasets import LIBRISPEECH

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=512,
    win_length=480, 
    hop_length=160,   
    n_mels=80
)

def collate_fn_pad(batch):
    spects = [t[0] for t in batch]
    segs = [t[1] for t in batch]
    labels = [t[2] for t in batch]
    lengths = [t[3] for t in batch]
    fnames = [t[4] for t in batch]

    padded_spects = torch.nn.utils.rnn.pad_sequence(spects, batch_first=True)
    lengths = torch.LongTensor(lengths)

    return padded_spects, segs, labels, lengths, fnames


def mfcc_size(wav_len, sr=16000, hop_length=160):
    return (wav_len - 1) // hop_length + 1

def get_subset(dataset, percent):
    A_split = int(len(dataset) * percent)
    B_split = len(dataset) - A_split
    dataset, _ = torch.utils.data.random_split(dataset, [A_split, B_split])
    return dataset

class WavPhnDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.data = list(iter_find_files(self.path, "*.wav"))
        super(WavPhnDataset, self).__init__()

    @staticmethod
    def get_datasets(path):
        raise NotImplementedError

    def process_file(self, wav_path):
        phn_path = wav_path.replace("wav", "phn")

        audio, sr = torchaudio.load(wav_path)
        audio = audio[0]
        mel_spect = mel_transform(audio).squeeze(0)
        log_mel = torch.log(mel_spect + 1e-8).transpose(0,1)
        # log_mel = (log_mel - log_mel.mean(dim=0, keepdim=True)) / (log_mel.std(dim=0, keepdim=True) + 1e-6)                       #Do not normalize log-mel as it smoothens out the transitions, making it hard to detect boundaries

        with open(phn_path, "r") as f:
            lines = f.readlines()
            lines = list(map(lambda line: line.split(" "), lines))

            times = torch.FloatTensor(list(map(lambda line: int(int(line[1]) / 160), lines)))[:-1]

            phonemes = list(map(lambda line: line[2].strip(), lines))

        return log_mel, times.tolist(), phonemes, wav_path

    def __getitem__(self, idx):
        log_mel, seg, phonemes, fname = self.process_file(self.data[idx])
        return log_mel, seg, phonemes, log_mel.size(0), fname

    def __len__(self):
        return len(self.data)


class TrainTestDataset(WavPhnDataset):
    def __init__(self, path):
        super(TrainTestDataset, self).__init__(path)

    @staticmethod
    def get_datasets(path, val_ratio=0.1):
        train_dataset = TrainTestDataset(join(path, 'train'))
        test_dataset  = TrainTestDataset(join(path, 'test'))

        train_len   = len(train_dataset)
        train_split = int(train_len * (1 - val_ratio))
        val_split   = train_len - train_split
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_split, val_split])

        train_dataset.path = join(path, 'train')
        val_dataset.path = join(path, 'train')

        return train_dataset, val_dataset, test_dataset


class TrainValTestDataset(WavPhnDataset):
    def __init__(self, paths):
        super(TrainValTestDataset, self).__init__(paths)

    @staticmethod
    def get_datasets(path, percent=1.0):
        train_dataset = TrainValTestDataset(join(path, 'train'))
        if percent != 1.0:
            train_dataset = get_subset(train_dataset, percent)
            train_dataset.path = join(path, 'train')
        val_dataset   = TrainValTestDataset(join(path, 'val'))
        test_dataset  = TrainValTestDataset(join(path, 'test'))

        return train_dataset, val_dataset, test_dataset


class LibriSpeechDataset(LIBRISPEECH):
    def __init__(self, path, subset, percent):
        self.libri_dataset = LIBRISPEECH(path, url=subset, download=False)
        if percent != 1.0:
            self.libri_dataset = get_subset(self.libri_dataset, percent)
        self.path = path
    
    def __getitem__(self, idx):
        wav, sr, utt, spk_id, chp_id, utt_id = self.libri_dataset[idx]
        wav = wav[0]
        hop_length = int(0.010 * sr)
        return wav, None, None, mfcc_size(len(wav), sr, hop_length), None

    def __len__(self):
        return len(self.libri_dataset)


class MixedDataset(Dataset):
    def __init__(self, ds1, ds2):
        self.ds1 = ds1
        self.ds2 = ds2
        self.path = f"{ds1.path}+{ds2.path}"
        self.ds1_len, self.ds2_len = len(ds1), len(ds2)
    
    def __len__(self):
        return self.ds1_len + self.ds2_len
    
    def __getitem__(self, idx):
        if idx < self.ds1_len:
            return self.ds1[idx]
        else:
            return self.ds2[idx - self.ds1_len]