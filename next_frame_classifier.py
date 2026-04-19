import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from utils import LambdaLayer, PrintShapeLayer, length_to_mask
from dataloader import TrainTestDataset                                                                 # Notations: Dissimilarity at timestep t (between d_t and d_(t + 1)) is represented as d_t
from collections import defaultdict
import torchaudio.transforms 
torch.set_printoptions(threshold=float('inf'))
import random
import sys

mae_loss = nn.L1Loss()

class NextFrameClassifier(nn.Module):
    def __init__(self, hp):
        super(NextFrameClassifier, self).__init__()
        self.hp = hp

        Z_DIM = hp.z_dim
        LS = hp.latent_dim if hp.latent_dim != 0 else Z_DIM

        self.enc = nn.GRU(                                                                              # A simple GRU encoder used on log-mel input in place of CNN to reduce smoothening effect introduced by the latter
            input_size=80,
            hidden_size=Z_DIM,
            num_layers=1,          
            batch_first=True,
            bidirectional=False
        )
        self.linear_proj = nn.Sequential(                                                               # A linear projection layer to project the 256-dim output of GRU encoder to 64-dim latent space for next frame classification (aka contrastive loss)
            nn.Dropout(0.1),
            nn.Linear(256,64)
        )
        self.seg_enc = nn.Sequential(                                                                   # A convolutional encoder that converts average of frames within a segment to a segmental representation with pentaphone context
            nn.Conv1d(Z_DIM, Z_DIM, kernel_size=5, padding=2)
        )
        self.mel_space =  nn.GRU(                                                                       # A GRU (to mirror the encoder) used to project the segment representations back to log-mel timescale for reconstruction of the input. The intuition is to give the emulated frame level representations some variation within a segment (the onset, nucleus and coda)
            input_size=Z_DIM,
            hidden_size=80,
            num_layers=1,          
            batch_first=True,
            bidirectional=False
        )           

        self.mel_out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(80,80)
        )                                                                                   

        self.pred_steps = list(range(1 + self.hp.pred_offset, 1 + self.hp.pred_offset + self.hp.pred_steps))
        print(f"prediction steps: {self.pred_steps}")
    
    def score(self, f, b):
        return F.cosine_similarity(f, b, dim=-1) * self.hp.cosine_coef

    def diff_boundary_detector(self, latent_vec, mask, thres=0.04):
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)                                                      
        sim = cos(latent_vec[:, :-1, :], latent_vec[:, 1:, :])                                          # Positive similarity (similarity between adjacent frames)
        d_mask = mask[:, :-1] * mask[:, 1:]                                                             # Creation of mask for padded similarity scores (from the positive similarity between the last unpadded frame and the first padded frame onwards in an utterance)   
        sim_min_masked = sim.masked_fill(~d_mask, float('inf'))                                         # As the 0s in the padded regime influence the min score needed for min-max normalization, we need to replace those padded 0s with inf to get true min
        sim_min = sim_min_masked.min(dim=1, keepdim=True).values                                        # Computing min for the normalization
        sim_max_masked = sim.masked_fill(~d_mask, float('-inf'))                                        # Similar explanation as above for this line and the next one
        sim_max = sim_max_masked.max(dim=1, keepdim=True).values
        d = 1 - ((sim - sim_min) / (sim_max - sim_min + 1e-6))                                          # Computation of dissimilarity as 1 - normalized_similarity
        d = d.masked_fill(~d_mask, 0)                                                                   # Filling the padded regime of dissimilarities with 0s
        zeros_col = torch.zeros(d.size(0), 1, dtype=torch.int32, device=latent_vec.device)              # Concatenation of a column of zeros to the right of a batch of dissimilarities 
        zeros_mask = torch.zeros(d.size(0), 1, dtype=torch.bool, device=latent_vec.device)
        d_mask = torch.cat([d_mask, zeros_mask], dim = 1)
        d = torch.cat([d, zeros_col], dim=1)                                                            # This is done to facilitate the computation of difference between d_t and d_(t - 1) [The left tensor need not include scores starting from timestep T (the last unpadded frame) as difference will be between 0 (first padded score) and d_T]
        left = torch.cat([zeros_col, d[:, :-1]], dim=1).masked_fill(~d_mask, 0)                          
        """                                                                                             For convenience, think of d as [ d_1 d_2 d_3 ... d_t     0   0 0...] which has T timesteps in total (including padding), 
                                                                                                                            left as    [  0  d_1 d_2 ... d_(t-1) d_t 0 0...] which also has T timesteps
                                                                                                                            right as   [ d_2 d_3 d_4 ... 0       0   0 0...] which also has T timesteps
                                                                                                                            left_2 as  [ 0    0  d_1 ... d_(t-2) ... 0 0...] which has T timesteps
                                                                                                                            right_2 as [ d_3 d_4 d_5 ... 0       0...0 0...] which has T timesteps
        """
        right = torch.cat([d[:, 1:], zeros_col], dim=1)                                                 # The same is done for obtaining the difference between d_t and d_(t + 1) [namely concatenating a column of zeros to the right of a batch of dissimilarity scores]
        zero = torch.tensor(0.0, device=latent_vec.device)                                              
        p_1 = torch.minimum(torch.maximum(d - left, zero), torch.maximum(d - right, zero))              # Choosing the min of the first order dissimilarity difference to the left and right (d - d_(t-1) and d - d_(t+1)). If the min difference between the adjacent dissimilarities is small, then dissimilarity at time t may not be significant compared to one of its first order neighbors to be considered a peak (they will be at the same level when visualized)
        left_2 = torch.cat([zeros_col, zeros_col, d[:, :-2]], dim=1).masked_fill(~d_mask, 0)            # Doing the same as above steps for second-order neighbors of dissimilarities
        right_2 = torch.cat([d[:, 2:], zeros_col, zeros_col], dim=1)            
        p_2 = torch.minimum(torch.maximum(d - left_2, zero), torch.maximum(d - right_2,  zero))         # Choosing the min of the second order dissimilarity difference to the left and right (d - d_(t-2) and d - d_(t+2)). Same intuition as in p_1.
        p = torch.minimum(torch.maximum(torch.maximum(p_1, p_2) - thres, zero), p_1)                    # We are choosing the max of min differences for the first-order and second-order case. p is the final list of decisions where values that are very high denotes a boundary event. Others are non-boundary events. If only p_1 was used to predict a boundary (by checking if the min difference of dissimilarities is more than a threshold), it may be a noisy boundary decision.
                                                                                                        # Now, we are consulting with p_2 as well. We take the max of p_1 and p_2 (max of min differences) and ask if that is atleast more than a threshold thres. If so, it is a phonetic boundary 
        p_soft = torch.tanh(10 * p)                                                                     # Differentiable component of the pred_boundaries variable introduced later
        p_hard = torch.tanh((10**9) * p)                                                                # Non-differentiable component of pred_boundaries that is marked as 1 for boundaries and 0 for non-boundary
        pred_boundaries = p_soft + (p_hard - p_soft).detach()                                           # End of the day, it is just p_hard (adding and subtracting p_soft so that in the backward pass, the soft component gets gradient updates and the hard component doesn't). These gradient updates aid in decreasing and increasing amplitudes of dissimilarities, thereby influencing the boundary decisions
                                                                                                        # It is not just amplitude shift of peaks (decrease and increase of amplitude) that happens due to backward pass. I believe time shift of peaks (aka boundaries) happens as decrease in amplitude of peak at some timestep t_1 and increase in amplitude at timestep t_2 due to the backward pass)
        return pred_boundaries, d

    def get_seg_rep(self, latent_vec, pred_boundaries, mask, scale=100000):                             
        T = pred_boundaries.size(1)
        seg_acc = torch.cumsum(pred_boundaries, dim=1).float().masked_fill(~mask,float('inf'))          # The cumulative sum as depicted in the paper
        seg_max = torch.max(seg_acc[mask.bool()])                                                       # This is done to find the maximum number of segments within a batch. As the implementation here is a batched version of the example in the paper, we need to handle it in this manner so that I do not end up creating separate Ms for each utterance in a batch
        M = int(seg_max.item()) + 1
        U = torch.arange(M, device=latent_vec.device).unsqueeze(1).expand(M,T)                          # Creating a batched version of the U matrix
        V = U.unsqueeze(0) - seg_acc.unsqueeze(1)                                                       # Same as the formulation in the paper from here onwards
        W_intermediate = 1 - torch.tanh(scale * torch.abs(V))                                                                                       
        durations = torch.sum(W_intermediate, dim=2, keepdim=True).to(torch.int64)                      # We can obtain durations of each segment by summing up rows in W (corresponds to columns in paper before making it to have unit sum)        
        W = W_intermediate / (durations + 1e-6)                                                         # Transposed version of W in the paper    
        seg_rep = torch.bmm(latent_vec.transpose(1,2), W.transpose(1,2))                                               
        return seg_rep.transpose(1,2), durations, V, W_intermediate

    def upsample(self, seg_rep, durations):
        B, M, D = seg_rep.shape                                                                         
        seg_rep_flatten = seg_rep.reshape(B*M,D)
        durations_flatten = durations.reshape(-1)
        upsampled_segs = torch.repeat_interleave(seg_rep_flatten, durations_flatten, dim=0)             # Using the durations obtained using get_seg_rep function to upsample the segment representations to match the number of frames in log-mel (It is mere repetition of segment representations to emulate the frames in log-mel)
        return upsampled_segs

    def frame_level_contrastive_loss(self, preds, padding_mask):                                        # Using contrastive loss as implemented in Kreuk et al 's paper
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
        gru_out, _ = self.enc(mel)
        latent_vec = self.linear_proj(gru_out)
        preds = defaultdict(list)
        for i, t in enumerate(self.pred_steps):  
            pos_pred = self.score(latent_vec[:, :-t], latent_vec[:, t:])  
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
                neg_pred = self.score(latent_vec[:, :-t], latent_vec[batch_reorder][: , time_reorder])  # This uses random permutation to choose negatives for any given anchor. The probability of choosing the anchor itself as negative is roughly 63%. On an average, you can expect one anchor to choose itself as negative. The impact of this needs to be studied.
                preds[t].append(neg_pred)
        pred_boundaries, d = self.diff_boundary_detector(gru_out, mask)
        seg_rep, durations, V, W_int = self.get_seg_rep(gru_out, pred_boundaries, mask)
        seg_rep = self.seg_enc(seg_rep.transpose(1,2))
        frame_level_rep = self.upsample(seg_rep.transpose(1,2), durations)
        reconstructed_mel, _ = self.mel_space(frame_level_rep)
        reconstructed_mel = self.mel_out(reconstructed_mel)
        return reconstructed_mel, frame_level_rep, seg_rep, durations, latent_vec, pred_boundaries, mask, d ,V, W_int, preds
    
    def loss(self, mask, input_mel, reconstructed_mel, preds):
        B,T,D = input_mel.shape
        reconstruction_loss = mae_loss(input_mel.reshape(B*T,D)[mask.reshape(-1)], reconstructed_mel)
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