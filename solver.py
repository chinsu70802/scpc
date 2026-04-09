import glob
from collections import OrderedDict, defaultdict

import dill
import wandb

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch_optimizer as optim_extra
from torch.optim.lr_scheduler import LambdaLR
import torch
from torch import optim
from torch.utils.data import ConcatDataset, DataLoader
import torchaudio

from dataloader import (LibriSpeechDataset, MixedDataset, TrainTestDataset,
                        TrainValTestDataset, collate_fn_pad)
from next_frame_classifier import NextFrameClassifier
from utils import (PrecisionRecallMetric, StatsMeter,
                   detect_peaks, line, max_min_norm, replicate_first_k_frames)


class Solver(LightningModule):
    def __init__(self, hparams):
        super(Solver, self).__init__()
        hp = hparams
        self.hp = hp
        self.hparams = hp
        self.peak_detection_params = defaultdict(lambda: {
            "prominence": 0.05,
            "width":      None,
            "distance":   None
        })
        self.pr = defaultdict(lambda: {
            "train": PrecisionRecallMetric(tolerance=2,mode='strict'),
            "val":   PrecisionRecallMetric(tolerance=2,mode='strict'),
            "test":  PrecisionRecallMetric(tolerance=2,mode='strict')
        })
        self.best_rval = defaultdict(lambda: {
            "train": (0, 0),
            "val":   (0, 0),
            "test":  (0, 0)
        })
        self.overall_best_rval = 0
        self.stats = defaultdict(lambda: {
            "train": StatsMeter(),
            "val":   StatsMeter(),
            "test":  StatsMeter()
        })

        wandb.init(project=self.hp.project, name=hp.exp_name, config=vars(hp), tags=[hp.tag])
        self.build_model()

    def prepare_data(self):
        # setup training set
        if "timit" in self.hp.data or "buckeye_1" in self.hp.data:
            train, val, test = TrainTestDataset.get_datasets(path=self.hp.timit_path)
        elif "buckeye" in self.hp.data:
            train, val, test = TrainValTestDataset.get_datasets(path=self.hp.buckeye_path, percent=self.hp.buckeye_percent)
        elif "telugu_slr" in self.hp.data:
            train, val, test = TrainTestDataset.get_datasets(path=self.hp.telugu_path)
        elif "hindi_slr" in self.hp.data:
            train, val, test = TrainTestDataset.get_datasets(path=self.hp.hindi_path)
        else:
            raise Exception("no such training data!")

        if "libri" in self.hp.data:
            libri_train = LibriSpeechDataset(path=self.hp.libri_path,
                                             subset=self.hp.libri_subset,
                                             percent=self.hp.libri_percent)
            train = ConcatDataset([train, libri_train])
            train.path = "\n\t+".join([dataset.path for dataset in train.datasets])
            print(f"added libri ({len(libri_train)} examples)")

        self.train_dataset = train
        self.valid_dataset = val
        self.test_dataset = test
        
        line()
        print("DATA:")
        print(f"train: {self.train_dataset.path} ({len(self.train_dataset)})")
        print(f"valid: {self.valid_dataset.path} ({len(self.valid_dataset)})")
        print(f"test: {self.test_dataset.path} ({len(self.test_dataset)})")
        line()
            
    def train_dataloader(self):
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.hp.batch_size,
                                       shuffle=True,
                                       collate_fn=collate_fn_pad,
                                       num_workers=self.hp.dataloader_n_workers)
        return self.train_loader

    def val_dataloader(self):
        self.valid_loader = DataLoader(self.valid_dataset,
                                       batch_size=self.hp.batch_size,
                                       shuffle=False,
                                       collate_fn=collate_fn_pad,
                                       num_workers=self.hp.dataloader_n_workers)
        return self.valid_loader

    def test_dataloader(self):
        self.test_loader  = DataLoader(self.test_dataset,
                                       batch_size=self.hp.batch_size,
                                       shuffle=False,
                                       collate_fn=collate_fn_pad,
                                       num_workers=self.hp.dataloader_n_workers)
        return self.test_loader

    def build_model(self):
        print("MODEL:")
        self.NFC = NextFrameClassifier(self.hp)
        line()

    def forward(self, data_batch, batch_i, mode):
        loss = 0

        # TRAIN
        log_mel, seg, phonemes, length, fname = data_batch
        reconstructed_mel, frame_level_rep, seg_rep, durations, latent_vec, pred_boundaries, mask, d, V, W_int, preds = self.NFC(log_mel, length)
        #NFC_loss = self.NFC.loss(preds, length)
        try:
            recons_loss, nce_loss = self.NFC.loss(mask, log_mel, reconstructed_mel, preds)
        except Exception as e:
            print(V.masked_fill(~mask.unsqueeze(1),float('inf')))
            print(W_int.masked_fill(~mask.unsqueeze(1),float('inf')))
            print(W_int.size(1))
            raise e
        NFC_loss = recons_loss + nce_loss
        self.stats['total'][mode].update(NFC_loss.item())
        self.stats['recons_loss'][mode].update(recons_loss.item())
        self.stats['nce_loss'][mode].update(nce_loss.item())
        loss += NFC_loss

        # INFERENCE
        if mode == "test" or (mode == "val" and self.hp.early_stop_metric == "val_max_rval"):
            positives = 0
            for t in self.NFC.pred_steps:
                p = d
                #p = replicate_first_k_frames(p, k=t, dim=1)
                positives += p
                #positives = 1 - max_min_norm(positives)
                self.pr[f'cpc_{t}'][mode].update(seg, positives, length)

        loss_key = "loss" if mode == "train" else f"{mode}_loss"
        return OrderedDict({
            loss_key: loss
        })

    def generic_eval_end(self, outputs, mode):
        metrics = {}
        data = self.hp.data

        for k, v in self.stats.items():
            metrics[f"train_{k}"] = self.stats[k]["train"].get_stats()
            metrics[f"{mode}_{k}"] = self.stats[k][mode].get_stats()

        epoch = self.current_epoch + 1
        metrics['epoch'] = epoch
        metrics['current_lr'] = self.opt.param_groups[0]['lr']

        line()
        for pred_type in self.pr.keys():
            if mode == "val":
                (precision, recall, f1, rval), (width, prominence, distance) = self.pr[pred_type][mode].get_stats()
                if rval > self.best_rval[pred_type][mode][0]:
                    self.best_rval[pred_type][mode] = rval, self.current_epoch
                    self.peak_detection_params[pred_type]["width"] = width
                    self.peak_detection_params[pred_type]["prominence"] = prominence
                    self.peak_detection_params[pred_type]["distance"] = distance
                    self.peak_detection_params[pred_type]["epoch"] = self.current_epoch
                    print(f"saving for test - {pred_type} - {self.peak_detection_params[pred_type]}")
            else:
                if data == 'buckeye_1':
                    (precision, recall, f1, rval), (width, prominence, distance) = self.pr[pred_type][mode].get_stats()
                else:
                    print(f"using pre-defined peak detection values - {pred_type} - {self.peak_detection_params[pred_type]}")
                    (precision, recall, f1, rval), _ = self.pr[pred_type][mode].get_stats(
                        width=self.peak_detection_params[pred_type]["width"],
                        prominence=self.peak_detection_params[pred_type]["prominence"],
                        distance=self.peak_detection_params[pred_type]["distance"],
                    )
                #
                # test has only one epoch so set it as best
                # this is to get the overall best pred_type later
                self.best_rval[pred_type][mode] = rval, self.current_epoch
            metrics[f'{data}_{mode}_{pred_type}_f1'] = f1
            metrics[f'{data}_{mode}_{pred_type}_precision'] = precision
            metrics[f'{data}_{mode}_{pred_type}_recall'] = recall
            metrics[f'{data}_{mode}_{pred_type}_rval'] = rval
            metrics[f"{data}_{mode}_{pred_type}_max_rval"] = self.best_rval[pred_type][mode][0]
            metrics[f"{data}_{mode}_{pred_type}_max_rval_epoch"] = self.best_rval[pred_type][mode][1]

        # get best rval from all rval types and all epochs
        best_overall_rval = -float("inf")
        for pred_type, rval in self.best_rval.items():
            if rval[mode][0] > best_overall_rval:
                best_overall_rval = rval[mode][0]
        metrics[f'{mode}_max_rval'] = best_overall_rval

        for k, v in metrics.items():
            print(f"\t{k:<30} -- {v}")
        line()
        wandb.log(metrics)

        output = OrderedDict({
            'log': metrics
        })

        return output

    def training_step(self, data_batch, batch_i):
        return self.forward(data_batch, batch_i, 'train')

    def validation_step(self, data_batch, batch_i):
        return self.forward(data_batch, batch_i, 'val')

    def test_step(self, data_batch, batch_i):
        return self.forward(data_batch, batch_i, 'test')

    def validation_end(self, outputs):
        return self.generic_eval_end(outputs, 'val')

    def test_end(self, outputs):
        return self.generic_eval_end(outputs, 'test')

    def configure_optimizers(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        if self.hp.optimizer == "sgd":
            self.opt = optim.SGD(parameters, lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4)
        elif self.hp.optimizer == "adam":
            self.opt = optim.Adam(parameters, lr=self.hparams.lr, weight_decay=5e-4)
        elif self.hp.optimizer == "ranger":
            self.opt = optim_extra.Ranger(parameters, lr=self.hparams.lr, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95, 0.999), eps=1e-5, weight_decay=0)
        else:
            raise Exception("unknown optimizer")
        print(f"optimizer: {self.opt}")
        line()
        self.scheduler = optim.lr_scheduler.StepLR(self.opt,
                                                   step_size=self.hp.lr_anneal_step,
                                                   gamma=self.hp.lr_anneal_gamma)
        return [self.opt]





    def on_epoch_end(self):
        self.scheduler.step()

    def on_save_checkpoint(self, ckpt):
        ckpt['peak_detection_params'] = dill.dumps(self.peak_detection_params)

    def on_load_checkpoint(self, ckpt):
        self.peak_detection_params = dill.loads(ckpt['peak_detection_params'])

    def get_ckpt_path(self):
        return glob.glob(self.hp.wd + "/*.ckpt")[0]


# import glob
# from collections import OrderedDict, defaultdict

# import dill
# import wandb

# from pytorch_lightning import LightningModule
# import torch_optimizer as optim_extra
# import torch
# from torch import optim
# from torch.utils.data import ConcatDataset, DataLoader

# from dataloader import (LibriSpeechDataset, TrainTestDataset,
#                         TrainValTestDataset, collate_fn_pad)
# from next_frame_classifier import NextFrameClassifier
# from utils import (PrecisionRecallMetric, StatsMeter,
#                    line, max_min_norm, replicate_first_k_frames)


# class Solver(LightningModule):


#     def __init__(self, hparams):
#         super(Solver, self).__init__()

#         # In PL 0.7.6 hparams must be set as a plain attribute
#         self.hparams = hparams
#         self.hp = hparams

#         # Peak detection parameters
#         self.peak_detection_params = defaultdict(lambda: {
#             "prominence": 0.05,
#             "width":      None,
#             "distance":   None
#         })

#         # Precision-Recall metrics
#         self.pr = defaultdict(lambda: {
#             "train": PrecisionRecallMetric(tolerance=2, mode='strict'),
#             "val":   PrecisionRecallMetric(tolerance=2, mode='strict'),
#             "test":  PrecisionRecallMetric(tolerance=2, mode='strict')
#         })

#         # Track best R-values
#         self.best_rval = defaultdict(lambda: {
#             "train": (0, 0),
#             "val":   (0, 0),
#             "test":  (0, 0)
#         })
#         self.overall_best_rval = 0

#         # Running statistics
#         self.stats = defaultdict(lambda: {
#             "train": StatsMeter(),
#             "val":   StatsMeter(),
#             "test":  StatsMeter()
#         })

#         wandb.init(
#             project=self.hp.project,
#             name=self.hp.exp_name,
#             config=vars(self.hp),
#             tags=[self.hp.tag]
#         )

#         self.build_model()

#     # ------------------------------------------------------------------
#     # Helpers
#     # ------------------------------------------------------------------

#     def _device(self):
#         """
#         Safe device resolution for PL 0.7.6 where self.device is unreliable.
#         """
#         try:
#             return next(self.parameters()).device
#         except StopIteration:
#             return torch.device('cpu')

#     # ------------------------------------------------------------------
#     # Model
#     # ------------------------------------------------------------------

#     def build_model(self):
#         print("MODEL:")
#         self.NFC = NextFrameClassifier(self.hp)
#         line()

#     # ------------------------------------------------------------------
#     # Data
#     # ------------------------------------------------------------------

#     def prepare_data(self):
#         if "timit" in self.hp.data or "buckeye_1" in self.hp.data:
#             train, val, test = TrainTestDataset.get_datasets(path=self.hp.timit_path)
#         elif "buckeye" in self.hp.data:
#             train, val, test = TrainValTestDataset.get_datasets(
#                 path=self.hp.buckeye_path,
#                 percent=self.hp.buckeye_percent
#             )
#         elif "telugu_slr" in self.hp.data:
#             train, val, test = TrainTestDataset.get_datasets(path=self.hp.telugu_path)
#         elif "hindi_slr" in self.hp.data:
#             train, val, test = TrainTestDataset.get_datasets(path=self.hp.hindi_path)
#         else:
#             raise Exception(f"Unknown training data: {self.hp.data}")

#         if "libri" in self.hp.data:
#             libri_train = LibriSpeechDataset(
#                 path=self.hp.libri_path,
#                 subset=self.hp.libri_subset,
#                 percent=self.hp.libri_percent
#             )
#             train = ConcatDataset([train, libri_train])
#             train.path = "\n\t+".join([d.path for d in train.datasets])
#             print(f"added libri ({len(libri_train)} examples)")

#         self.train_dataset = train
#         self.valid_dataset = val
#         self.test_dataset  = test

#         line()
#         print("DATA:")
#         print(f"train: {self.train_dataset.path} ({len(self.train_dataset)})")
#         print(f"valid: {self.valid_dataset.path} ({len(self.valid_dataset)})")
#         print(f"test:  {self.test_dataset.path}  ({len(self.test_dataset)})")
#         line()

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.hp.batch_size,
#             shuffle=True,
#             collate_fn=collate_fn_pad,
#             num_workers=self.hp.dataloader_n_workers
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.valid_dataset,
#             batch_size=self.hp.batch_size,
#             shuffle=False,
#             collate_fn=collate_fn_pad,
#             num_workers=self.hp.dataloader_n_workers
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.hp.batch_size,
#             shuffle=False,
#             collate_fn=collate_fn_pad,
#             num_workers=self.hp.dataloader_n_workers
#         )

#     def configure_optimizers(self):
#         params = [p for p in self.parameters() if p.requires_grad]

#         print("Num trainable params:", len(params))  # debug

#         if len(params) == 0:
#             raise ValueError("No parameters found!")
#         params = list(self.parameters())

#         print("DEBUG: num params =", len(params))  # MUST be > 0

#         self.opt = optim.Adam(
#             params,
#             lr=self.hp.lr,
#             weight_decay=5e-4
#         )

#         return [self.opt]
#     # ------------------------------------------------------------------
#     # Forward / steps
#     # ------------------------------------------------------------------

#     def forward(self, data_batch, batch_i, mode):
#         """
#         Forward pass: L = nce_weight * L_NFC + recon_weight * L_recon
#         """
#         device = next(self.parameters()).device

#         try:
#             log_mel, seg, phonemes, length, fname = data_batch

#             output    = self.NFC(log_mel, length)
#             loss_dict = self.NFC.loss(output)

#             total_loss = loss_dict['total']
#             nfc_loss   = loss_dict['nfc']
#             recon_loss = loss_dict['recon']

#             # Running statistics
#             self.stats['nfc_loss'][mode].update(nfc_loss.item())
#             self.stats['recon_loss'][mode].update(recon_loss.item())
#             self.stats['total_loss'][mode].update(total_loss.item())

#             # Step-level wandb logging
#             wandb.log({
#                 f"{mode}_total_loss": total_loss.item(),
#                 f"{mode}_nfc_loss":   nfc_loss.item(),
#                 f"{mode}_recon_loss": recon_loss.item(),
#             })

#             loss = total_loss

#             # Segmentation metrics (val / test only)
#             if mode == "test" or (
#                 mode == "val" and self.hp.early_stop_metric == "val_max_rval"
#             ):
#                 p = output.get('d')   # [B, T-1] dissimilarity scores
#                 if p is not None and seg is not None and length is not None:
#                     try:
#                         self.pr['phoneme'][mode].update(seg, p, length)
#                     except Exception as e:
#                         print(f"WARNING: Could not update PR metrics: {e}")

#         except Exception as e:
#             import traceback
#             print(f"ERROR in forward pass ({mode}): {e}")
#             traceback.print_exc()
#             # Differentiable fallback so PL training loop does not crash
#             loss = torch.tensor(1.0, device=device, requires_grad=True)

#         loss_key = "loss" if mode == "train" else f"{mode}_loss"
#         return OrderedDict({loss_key: loss})

#     def training_step(self, data_batch, batch_i):
#         return self.forward(data_batch, batch_i, 'train')

#     def validation_step(self, data_batch, batch_i):
#         return self.forward(data_batch, batch_i, 'val')

#     def test_step(self, data_batch, batch_i):
#         return self.forward(data_batch, batch_i, 'test')

#     # ------------------------------------------------------------------
#     # Epoch-end evaluation  (PL 0.7.6 hook names)
#     # ------------------------------------------------------------------

#     def generic_eval_end(self, outputs, mode):
#         metrics = {}
#         data    = self.hp.data

#         # Aggregate running stats
#         try:
#             for k in self.stats.keys():
#                 metrics[f"train_{k}"] = self.stats[k]["train"].get_stats()
#                 metrics[f"{mode}_{k}"] = self.stats[k][mode].get_stats()
#         except Exception as e:
#             print(f"WARNING: Could not aggregate stats: {e}")

#         metrics['epoch'] = self.current_epoch + 1
#         if hasattr(self, 'opt') and self.opt is not None:
#             metrics['current_lr'] = self.opt.param_groups[0]['lr']

#         line()

#         epoch_log_dict = {}
#         for loss_key in ['nfc_loss', 'recon_loss', 'total_loss']:
#             if loss_key in self.stats:
#                 try:
#                     v = self.stats[loss_key][mode].get_stats()
#                     epoch_log_dict[f"epoch_{mode}_{loss_key}"] = v
#                     metrics[f"{mode}_{loss_key}_epoch"]         = v
#                 except Exception:
#                     pass

#         # Segmentation metrics
#         for pred_type in list(self.pr.keys()):
#             try:
#                 if mode == "val":
#                     (precision, recall, f1, rval), (width, prominence, distance) = \
#                         self.pr[pred_type][mode].get_stats()
#                     if rval > self.best_rval[pred_type][mode][0]:
#                         self.best_rval[pred_type][mode] = (rval, self.current_epoch)
#                         self.peak_detection_params[pred_type].update({
#                             "width":      width,
#                             "prominence": prominence,
#                             "distance":   distance,
#                             "epoch":      self.current_epoch
#                         })
#                         print(f"saving for test - {pred_type} - "
#                               f"{self.peak_detection_params[pred_type]}")
#                 else:
#                     if data == 'buckeye_1':
#                         (precision, recall, f1, rval), _ = \
#                             self.pr[pred_type][mode].get_stats()
#                     else:
#                         print(f"using pre-defined peak detection - {pred_type} - "
#                               f"{self.peak_detection_params[pred_type]}")
#                         (precision, recall, f1, rval), _ = \
#                             self.pr[pred_type][mode].get_stats(
#                                 width=self.peak_detection_params[pred_type]["width"],
#                                 prominence=self.peak_detection_params[pred_type]["prominence"],
#                                 distance=self.peak_detection_params[pred_type]["distance"],
#                             )
#                     self.best_rval[pred_type][mode] = (rval, self.current_epoch)

#                 metrics[f'{data}_{mode}_{pred_type}_f1']            = f1
#                 metrics[f'{data}_{mode}_{pred_type}_precision']      = precision
#                 metrics[f'{data}_{mode}_{pred_type}_recall']         = recall
#                 metrics[f'{data}_{mode}_{pred_type}_rval']           = rval
#                 metrics[f"{data}_{mode}_{pred_type}_max_rval"]       = self.best_rval[pred_type][mode][0]
#                 metrics[f"{data}_{mode}_{pred_type}_max_rval_epoch"] = self.best_rval[pred_type][mode][1]
#             except Exception as e:
#                 print(f"WARNING: Could not compute metrics for {pred_type}: {e}")

#         best_overall = max(
#             (v[mode][0] for v in self.best_rval.values()),
#             default=-float("inf")
#         )
#         metrics[f'{mode}_max_rval'] = best_overall

#         for k, v in metrics.items():
#             print(f"\t{k:<40} -- {v}")
#         line()

#         try:
#             wandb.log({**metrics, **epoch_log_dict})
#         except Exception as e:
#             print(f"WARNING: wandb log failed: {e}")

#         return OrderedDict({'log': metrics})

#     # PL 0.7.6 hook names
#     def validation_end(self, outputs):
#         return self.generic_eval_end(outputs, 'val')

#     def test_end(self, outputs):
#         return self.generic_eval_end(outputs, 'test')

#     def on_epoch_end(self):
#         """Step LR scheduler manually — required in PL 0.7.6."""
#         if hasattr(self, 'scheduler') and self.scheduler is not None:
#             self.scheduler.step()

#     def grad_norm(self, norm_type):
#         results    = {}
#         total_norm = torch.tensor(0.0)
#         for name, p in self.named_parameters():
#             if p.requires_grad and p.grad is not None:
#                 try:
#                     param_norm  = p.grad.data.norm(norm_type)
#                     total_norm += param_norm.item() ** norm_type
#                     results[f'grad_{norm_type}_norm_{name}'] = round(param_norm.item(), 3)
#                 except Exception:
#                     pass
#         total_norm = total_norm.item() ** (1.0 / norm_type)
#         results[f'grad_{norm_type}_norm_total'] = round(total_norm, 3)
#         return results

#     def on_save_checkpoint(self, ckpt):
#         ckpt['peak_detection_params'] = dill.dumps(self.peak_detection_params)

#     def on_load_checkpoint(self, ckpt):
#         self.peak_detection_params = dill.loads(ckpt['peak_detection_params'])

#     def get_ckpt_path(self):
#         return glob.glob(self.hp.wd + "/*.ckpt")[0]