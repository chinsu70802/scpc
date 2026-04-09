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

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import hydra
# from utils import LambdaLayer, PrintShapeLayer, length_to_mask
# from dataloader import TrainTestDataset
# from collections import defaultdict
# import torchaudio.transforms

# class NextFrameClassifier(nn.Module):
    
#     def __init__(self, hp):
#         super(NextFrameClassifier, self).__init__()
#         self.hp = hp

#         self.conv = nn.Sequential(
#             nn.Conv1d(80, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv1d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU()
#         )
#         self.linear_proj = nn.Linear(256, 64)
        

#         self.seg_enc = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256)
#         )
        
#         self.temporal_autoencoder = nn.Sequential(
#             nn.Linear(256, 64),
#             nn.ReLU(),
#             nn.Linear(64, 80)
#         )
        
#         self.recon_weight = getattr(hp, 'recon_weight', 1.0) 
#         self.nce_weight = getattr(hp, 'nce_weight', 1.0)

#     def diff_boundary_detector(self, latent_vec, mask, thres=0.04):
#         cos = nn.CosineSimilarity(dim=2, eps=1e-6)
#         sim = cos(latent_vec[:, :-1, :], latent_vec[:, 1:, :])
#         d_mask = mask[:, :-1] & mask[:, 1:]
#         sim_min_masked = sim.masked_fill(~d_mask, float('inf'))
#         sim_min = sim_min_masked.min(dim=1, keepdim=True).values
#         sim_max_masked = sim.masked_fill(~d_mask, float('-inf'))
#         sim_max = sim_max_masked.max(dim=1, keepdim=True).values
        
#         d = 1 - ((sim - sim_min) / (sim_max - sim_min + 1e-6))
#         d = d.masked_fill(~d_mask, 0)
        
#         zeros_col = torch.zeros(d.size(0), 1, device=latent_vec.device)
#         d = torch.cat([zeros_col, d], dim=1)  # [B, T]
        
#         zero = torch.zeros_like(d)
#         left = torch.cat([zeros_col, d[:, :-1]], dim=1).masked_fill(~mask, 0)
#         right = torch.cat([d[:, 1:], zeros_col], dim=1)
#         right[:, 0] = 0.0
        
#         p_1 = torch.minimum(
#             torch.maximum(d - left, zero),
#             torch.maximum(d - right, zero)
#         )
        
#         left_2 = torch.cat([zeros_col, zeros_col, d[:, :-2]], dim=1).masked_fill(~mask, 0)
#         right_2 = torch.cat([d[:, 2:], zeros_col, zeros_col], dim=1)
#         right_2[:, 0] = 0.0
        
#         p_2 = torch.minimum(
#             torch.maximum(d - left_2, zero),
#             torch.maximum(d - right_2, zero)
#         )
        
#         p = torch.minimum(
#             torch.maximum(torch.maximum(p_1, p_2) - thres, zero),
#             p_1
#         )
        
#         p_soft = torch.tanh(10 * p)
#         p_hard = torch.tanh(1000 * p)
#         pred_boundaries = p_soft + (p_hard - p_soft).detach()
        
#         return pred_boundaries, d

#     def get_seg_rep(self, latent_vec, pred_boundaries, scale=1000):
#         B, T, D = latent_vec.shape
        
#         seg_acc = torch.cumsum(pred_boundaries, dim=1)  # [B, T]
#         seg_max = torch.max(seg_acc[:, -1].unsqueeze(1), dim=0, keepdim=True).values
#         M = max(1, int(torch.ceil(seg_max).item()))  # Number of segments
        
#         U = torch.arange(M, device=latent_vec.device).unsqueeze(1).expand(int(M), int(T))
#         V = U.unsqueeze(0) - seg_acc.unsqueeze(1)  # [B, M, T]
        
#         W_intermediate = 1 - torch.tanh(scale * torch.abs(V))  # [B, M, T]
        
#         # Normalize weights per segment
#         durations = torch.sum(W_intermediate, dim=2, keepdim=True).squeeze(-1)  # [B, M]
#         durations = durations / (durations.sum(dim=1, keepdim=True) + 1e-6)
#         durations = durations * T  # Scale to sequence length (stays CONTINUOUS!)
        
#         # Compute final weights
#         W = W_intermediate / (durations.unsqueeze(-1) + 1e-6)  # [B, M, T]
        
#         # Segment representation: weighted average of frame latents
#         seg_rep = latent_vec.transpose(1, 2) @ W.transpose(1, 2)  # [B, D, M]
#         seg_rep = seg_rep.transpose(1, 2)  # [B, M, D]
        
#         # Apply segment encoder
#         seg_rep = self.seg_enc(seg_rep)  # [B, M, D]
        
#         return seg_rep, durations  # durations are CONTINUOUS!

#     def upsample(self, seg_rep, durations, target_length=None):
#         B, M, D = seg_rep.shape
        
#         upsampled_list = []
#         for b in range(B):
#             seg_rep_b = seg_rep[b] 
#             durations_b = durations[b]  
            
#             if M > 0 and durations_b.sum() > 0:
#                 durations_int = torch.round(durations_b).long()
                
#                 # If target_length specified, adjust last segment
#                 if target_length is not None:
#                     T_b = target_length
#                 else:
#                     T_b = int(durations_int.sum().item())
                
#                 diff = T_b - durations_int.sum().item()
#                 if diff != 0:
#                     durations_int[-1] += int(diff)
                
#                 durations_int = torch.clamp(durations_int, min=0)
                
#                 if durations_int.sum() > 0:
#                     upsampled_b = torch.repeat_interleave(seg_rep_b, durations_int, dim=0)
#                 else:
#                     upsampled_b = torch.zeros(1, D, device=seg_rep.device)
#             else:
#                 upsampled_b = torch.zeros(1, D, device=seg_rep.device)
            
#             upsampled_list.append(upsampled_b)
        
#         return upsampled_list

#     def frame_level_contrastive_loss(self, latent_vec, mask, K=3):

#         B, T, D = latent_vec.shape
#         device = latent_vec.device

#         latent_vec = F.normalize(latent_vec, dim=-1)
        
#         # Anchor: z_t, Positive: z_{t+1}
#         anchor = latent_vec[:, :-1, :]  # [B, T-1, D]
#         positive = latent_vec[:, 1:, :]  # [B, T-1, D]
        
#         cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
#         pos_sim = cos(anchor, positive) / 0.1  # [B, T-1]
        
#         # Sample K negative examples per anchor
#         perm = torch.rand(B, T-1, T, device=device).argsort(dim=-1)  # [B, T-1, T]
        
#         # Create masks for invalid indices
#         t_idx = torch.arange(T-1, device=device).view(1, T-1, 1).expand(B, T-1, T)
        
#         # Invalid: current frame (t) and next frame (t+1) - NO wraparound
#         invalid_current = (perm == t_idx)
#         invalid_next = (perm == (t_idx + 1))
#         valid_mask = ~(invalid_current | invalid_next)
        
#         # Also respect padding mask
#         frame_valid = mask[:, :T]  # [B, T]
#         frame_valid_exp = frame_valid.unsqueeze(1).expand(B, T-1, T)
#         frame_valid_neg = torch.gather(frame_valid_exp, 2, perm)

#         valid_mask = valid_mask & frame_valid_neg

#         # Mask invalid indices
#         perm_masked = perm.clone()
#         perm_masked[~valid_mask] = T + 1

#         # Sort and select K
#         perm_sorted, _ = perm_masked.sort(dim=-1)
#         neg_idx = perm_sorted[:, :, :K]

#         # Clamp to safe range
#         neg_idx = neg_idx.clamp(max=T-1)

#         # Gather negatives
#         latent_exp = latent_vec.unsqueeze(1).expand(B, T-1, T, D)
#         neg_idx_exp = neg_idx.unsqueeze(-1).expand(-1, -1, -1, D)

#         negatives = torch.gather(latent_exp, 2, neg_idx_exp)
        
#         # Compute similarities for negatives
#         anchor_exp = anchor.unsqueeze(2)  # [B, T-1, 1, D]
#         neg_sim = cos(anchor_exp, negatives) / 0.1  # [B, T-1, K]
        
#         # Stack positive and negatives for cross-entropy
#         logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # [B, T-1, K+1]
#         labels = torch.zeros(B, T-1, dtype=torch.long, device=device)
        
#         # Flatten and filter by valid anchors
#         logits = logits.reshape(-1, K+1)
#         labels = labels.reshape(-1)
#         valid_anchors = (mask[:, :-1] & mask[:, 1:]).reshape(-1).bool()
        
#         logits = logits[valid_anchors]
#         labels = labels[valid_anchors]
        
#         if logits.shape[0] > 0:
#             loss = F.cross_entropy(logits, labels)
#         else:
#             loss = torch.tensor(0.0, device=device, requires_grad=True)
        
#         return loss
    
#     def forward(self, mel, lengths):

#         B, T, _ = mel.shape
        
#         # Handle lengths
#         if isinstance(lengths, list):
#             lengths_tensor = torch.tensor(lengths, device=mel.device, dtype=torch.long)
#         elif isinstance(lengths, torch.Tensor):
#             lengths_tensor = lengths.to(mel.device).long()
#         else:
#             lengths_tensor = lengths
        
#         # Create mask
#         mask = torch.arange(T, device=mel.device).unsqueeze(0) < lengths_tensor.unsqueeze(1)
        
#         # Frame encoding
#         conv_out = self.conv(mel.transpose(1, 2))  # [B, 256, T]
#         latent_vec = self.linear_proj(conv_out.transpose(1, 2))  # [B, T, 64]
        
#         # Boundary detection
#         pred_boundaries, d = self.diff_boundary_detector(conv_out.transpose(1,2), mask)
        
#         # Segment extraction
#         seg_rep, durations = self.get_seg_rep(conv_out.transpose(1,2), pred_boundaries)
        
#         # Upsample back to frame level
#         upsampled_list = self.upsample(seg_rep, durations, target_length=T)
        
#         # Reconstruct mel-spectrogram from upsampled segments
#         reconstructed_mel_list = []
#         for b in range(B):
#             upsampled_b = upsampled_list[b]  # [T_b, 64]
#             valid_len = int(lengths_tensor[b].item())
            
#             # Reconstruct only valid frames
#             if valid_len > 0 and upsampled_b.shape[0] > 0:
#                 upsampled_valid = upsampled_b[:min(valid_len, upsampled_b.shape[0])]
#                 reconstructed_b = self.temporal_autoencoder(upsampled_valid)  # [T_valid, 80]
#                 reconstructed_mel_list.append(reconstructed_b)
#             else:
#                 # Empty sequence
#                 reconstructed_mel_list.append(torch.zeros(1, 80, device=mel.device))
        
#         return {
#             'latent_vec': latent_vec,
#             'seg_rep': seg_rep,
#             'durations': durations,
#             'pred_boundaries': pred_boundaries,
#             'd': d,
#             'mask': mask,
#             'upsampled': upsampled_list,
#             'reconstructed_mel_list': reconstructed_mel_list,
#             'mel': mel,
#             'lengths': lengths_tensor
#         }
    
#     def loss(self, output_dict):

#         latent_vec = output_dict['latent_vec']
#         mask = output_dict['mask']
#         reconstructed_mel_list = output_dict['reconstructed_mel_list']
#         mel = output_dict['mel']
#         lengths = output_dict['lengths']
        
#         B, T, D = mel.shape
#         device = mel.device
        
#         # 1. Frame-level contrastive loss
#         nfc_loss = self.frame_level_contrastive_loss(latent_vec, mask)
        
#         # 2. Reconstruction loss on log-mel
#         recon_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
#         if len(reconstructed_mel_list) > 0:
#             recon_losses = []
#             for b in range(B):
#                 valid_len = int(lengths[b].item())
#                 if valid_len > 0:
#                     # Original mel frames
#                     mel_valid = mel[b, :valid_len, :]  # [T_b, 80]
                    
#                     # Reconstructed mel
#                     reconstructed_b = reconstructed_mel_list[b]  # [T_b, 80]
                    
#                     # Align lengths if necessary
#                     min_len = min(mel_valid.shape[0], reconstructed_b.shape[0])
#                     mel_valid = mel_valid[:min_len]
#                     reconstructed_b = reconstructed_b[:min_len]
                    
#                     if min_len > 0:
#                         loss_b = F.mse_loss(mel_valid, reconstructed_b)
#                         recon_losses.append(loss_b)
            
#             if recon_losses:
#                 recon_loss = torch.stack(recon_losses).mean()
        
#         # Check for NaN/Inf
#         if torch.isnan(nfc_loss) or torch.isinf(nfc_loss):
#             nfc_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
#         if torch.isnan(recon_loss) or torch.isinf(recon_loss):
#             recon_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
#         # Clamp to avoid explosion
#         nfc_loss = torch.clamp(nfc_loss, min=0, max=1e6)
#         recon_loss = torch.clamp(recon_loss, min=0, max=1e6)
        
#         # Total loss
#         total_loss = self.nce_weight * nfc_loss + self.recon_weight * recon_loss
        
#         return {
#             'total': total_loss,
#             'nfc': nfc_loss,
#             'recon': recon_loss
#         }


# @hydra.main(config_path='conf/config.yaml', strict=False)
# def main(cfg):
#     ds, _, _ = TrainTestDataset.get_datasets(cfg.timit_path)
#     spect, seg, phonemes, length, fname = ds[0]
#     spect = spect.unsqueeze(0)

#     model = NextFrameClassifier(cfg)
#     output = model(spect, [length])
#     loss_dict = model.loss(output)
    
#     print(f"Latent shape: {output['latent_vec'].shape}")
#     print(f"Segment rep shape: {output['seg_rep'].shape}")
#     print(f"Durations shape: {output['durations'].shape}")
#     print(f"Total loss: {loss_dict['total'].item():.4f}")
#     print(f"  NFC loss: {loss_dict['nfc'].item():.4f}")
#     print(f"  Reconstruction loss: {loss_dict['recon'].item():.4f}")


# if __name__ == "__main__":
#     main()