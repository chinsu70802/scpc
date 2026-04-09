import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from utils import LambdaLayer, PrintShapeLayer, length_to_mask
from dataloader import TrainTestDataset
from collections import defaultdict
import torchaudio.transforms 
torch.set_printoptions(threshold=float('inf'))
import random
import sys

mse_loss = nn.MSELoss()

class NextFrameClassifier(nn.Module):
    def __init__(self, hp):
        super(NextFrameClassifier, self).__init__()
        self.hp = hp

        Z_DIM = hp.z_dim
        LS = hp.latent_dim if hp.latent_dim != 0 else Z_DIM

        self.conv = nn.Sequential(
            nn.Conv1d(80,256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256,256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.linear_proj = nn.Linear(256,64)
        self.seg_enc = nn.Sequential(
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,64)
        )
        self.temporal_autoencoder = nn.Sequential(
            nn.Linear(64,32), # I guess the capacity should be high!
            nn.ReLU(),
            nn.Linear(32,80)
        )
        self.pred_steps = list(range(1 + self.hp.pred_offset, 1 + self.hp.pred_offset + self.hp.pred_steps))
        print(f"prediction steps: {self.pred_steps}")
    
    def score(self, f, b):
        return F.cosine_similarity(f, b, dim=-1) * self.hp.cosine_coef

    def diff_boundary_detector(self, latent_vec, mask, thres=0.04):
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        sim = cos(latent_vec[:, :-1, :], latent_vec[:, 1:, :])
        d_mask = mask[:, :-1] * mask[:, 1:]
        sim_min_masked = sim.masked_fill(~d_mask, float('inf'))
        sim_min = sim_min_masked.min(dim=1, keepdim=True).values
        sim_max_masked = sim.masked_fill(~d_mask, float('-inf'))
        sim_max = sim_max_masked.max(dim=1, keepdim=True).values
        d = 1 - ((sim - sim_min) / (sim_max - sim_min + 1e-6))
        d = d.masked_fill(~d_mask, 0)
        zeros_col = torch.zeros(d.size(0), 1, device=latent_vec.device)
        d = torch.cat([zeros_col, d], dim=1)
        left = torch.cat([zeros_col, d[:, :-1]], dim=1).masked_fill(~mask, 0)
        right = torch.cat([d[:, 1:], zeros_col], dim=1)
        right[:, 0] = 0.0
        zero = torch.tensor(0.0, device=latent_vec.device)
        p_1 = torch.minimum(torch.maximum(d - left, zero), torch.maximum(d - right, zero))
        left_2 = torch.cat([zeros_col, zeros_col, d[:, :-2]], dim=1).masked_fill(~mask, 0)
        right_2 = torch.cat([d[:, 2:], zeros_col, zeros_col], dim=1)
        right_2[:, 0] = 0.0
        p_2 = torch.minimum(torch.maximum(d - left_2, zero), torch.maximum(d - right_2,  zero))
        p = torch.minimum(torch.maximum(torch.maximum(p_1, p_2) - thres, zero), p_1)
        p_soft = torch.tanh(10 * p)
        p_hard = torch.tanh((10**9) * p)
        pred_boundaries = p_soft + (p_hard - p_soft).detach()
        return pred_boundaries, d

    def get_seg_rep(self, latent_vec, pred_boundaries, mask, scale=100000):
        T = pred_boundaries.size(1)
        seg_acc = torch.cumsum(pred_boundaries, dim=1).float().masked_fill(~mask,float('inf'))
        seg_max = torch.max(seg_acc[mask.bool()])
        M = int(seg_max.item()) + 1
        U = torch.arange(M, device=latent_vec.device).unsqueeze(1).expand(M,T)
        V = U.unsqueeze(0) - seg_acc.unsqueeze(1)
        W_intermediate = 1 - torch.tanh(scale * torch.abs(V))
        durations = torch.sum(W_intermediate, dim=2, keepdim=True).to(torch.int64)
        W = W_intermediate / (durations + 1e-6)
        seg_rep = torch.bmm(latent_vec.transpose(1,2), W.transpose(1,2))
        return seg_rep.transpose(1,2), durations, V, W_intermediate

    def upsample(self, seg_rep, durations):
        B, M, D = seg_rep.shape
        seg_rep_flatten = seg_rep.reshape(B*M,D)
        durations_flatten = durations.reshape(-1)
        upsampled_segs = torch.repeat_interleave(seg_rep_flatten, durations_flatten, dim=0)
        return upsampled_segs

    def frame_level_contrastive_loss(self, preds, padding_mask):
        loss = 0
        mask = padding_mask[:,:-1] * padding_mask[:,1:]
        for t, t_preds in preds.items():
            out = torch.stack(t_preds, dim=-1)
            out = F.log_softmax(out, dim=-1)
            out = out[...,0] * mask
            loss += -out.mean()
        return loss
    
    def forward(self,mel, lengths):
        B,T,_ = mel.shape
        mask = torch.arange(T, device=mel.device).unsqueeze(0) < torch.tensor(lengths, device=mel.device).unsqueeze(1)
        conv_out = self.conv(mel.transpose(1,2))
        latent_vec = self.linear_proj(conv_out.transpose(1,2))
        preds = defaultdict(list)
        for i, t in enumerate(self.pred_steps):  # predict for steps 1...t
            pos_pred = self.score(latent_vec[:, :-t], latent_vec[:, t:])  # score for positive frame
            preds[t].append(pos_pred)

            for _ in range(self.hp.n_negatives):
                if self.training:
                    time_reorder = torch.randperm(pos_pred.shape[1])
                    batch_reorder = torch.arange(pos_pred.shape[0])
                    if self.hp.batch_shuffle:
                        batch_reorder = torch.randperm(pos_pred.shape[0])
                else:
                    time_reorder = torch.arange(pos_pred.shape[1])
                    batch_reorder = torch.arange(pos_pred.shape[0])
                neg_pred = self.score(latent_vec[:, :-t], latent_vec[batch_reorder][: , time_reorder])  # score for negative random frame
                preds[t].append(neg_pred)
        pred_boundaries, d = self.diff_boundary_detector(latent_vec, mask)
        seg_rep, durations, V, W_int = self.get_seg_rep(latent_vec, pred_boundaries, mask)
        seg_rep = self.seg_enc(seg_rep)
        frame_level_rep = self.upsample(seg_rep, durations)
        reconstructed_mel = self.temporal_autoencoder(frame_level_rep)
        return reconstructed_mel, frame_level_rep, seg_rep, durations, latent_vec, pred_boundaries, mask, d ,V, W_int, preds
    
    def loss(self, mask, input_mel, reconstructed_mel, preds):
        B,T,D = input_mel.shape
        reconstruction_loss = mse_loss(input_mel.reshape(B*T,D)[mask.reshape(-1)], reconstructed_mel)
        nce_loss = self.frame_level_contrastive_loss(preds, mask)
        return reconstruction_loss, nce_loss

@hydra.main(config_path='conf/config.yaml', strict=False)
def main(cfg):
    ds, _, _ = TrainTestDataset.get_datasets(cfg.timit_path)
    spect, seg, phonemes, length, fname = ds[0]
    spect = spect.unsqueeze(0)

    model = NextFrameClassifier(cfg)
    out = model(spect, length)

if __name__ == "__main__":
    main()