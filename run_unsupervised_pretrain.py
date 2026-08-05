import os
import argparse
import pickle
import json
import hashlib
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# from model import UnsupervisedPretrain
from model.dwmoespace_newgate import UnsupervisedPretrain
from utils import UnsupervisedPretrainLoader, collate_fn_unsupervised_pretrain
from utils import TwoDatasetPretrainLoader, collate_fn_two_dataset_pretrain
from experiment_utils import artifact_metadata, save_experiment_result, seed_everything
# from utils import TUABTUEVCHBMITLoader,collate_fn_tuabtuevchbmit_datasets #SEED

     
class LitModel_supervised_pretrain(pl.LightningModule):
    def __init__(self, args, save_path):
        super().__init__()
        self.args = args
        self.save_path = save_path
        self.T = 0.2
        self.model = UnsupervisedPretrain(
            num_channels=16,
            compressed_time=args.feature_out,
            expert_output_dim=args.feature_out,
            need_token=args.need_token,
            raw_time=args.sampling_rate * args.window_seconds,
            gate_type=args.gate_type,
            gate_hidden_dim=args.gate_hidden_dim,
            expert_axis=args.expert_axis,
            relation_mode=args.relation_mode,
            norm_type=args.norm_type,
            gate_descriptor_source=args.gate_descriptor_source,
            match_dynamic_budget=args.match_dynamic_budget,
            expert_kernels=args.expert_kernels,
            objective=args.objective,
        )
        if args.objective in (
            "normalized_masked_stats",
            "shuffled_masked_stats",
        ):
            if not args.stat_normalization:
                raise ValueError(
                    f"{args.objective} requires --stat_normalization"
                )
            normalization = json.loads(
                Path(args.stat_normalization).read_text(encoding="utf-8")
            )
            expected_metadata = {
                "logdataset": args.logdataset,
                "sampling_rate_hz": args.sampling_rate,
                "window_seconds": args.window_seconds,
                "window_samples": args.sampling_rate * args.window_seconds,
                "segment_count": args.need_token,
            }
            mismatches = {
                key: {"artifact": normalization.get(key), "requested": value}
                for key, value in expected_metadata.items()
                if normalization.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    "Statistic normalization metadata mismatch: "
                    + json.dumps(mismatches, sort_keys=True)
                )
            expected = self.model.stats_features
            if normalization.get("stats_features") != expected:
                raise ValueError(
                    "Statistic normalization feature order differs from model: "
                    f"{normalization.get('stats_features')} vs {expected}"
                )
            means = torch.tensor(normalization["mean"], dtype=torch.float32)
            stds = torch.tensor(normalization["std"], dtype=torch.float32)
            if means.shape != (self.model.num_stats,) or stds.shape != means.shape:
                raise ValueError("Statistic normalization has an invalid shape")
            if not torch.isfinite(means).all() or not torch.isfinite(stds).all():
                raise ValueError("Statistic normalization contains non-finite values")
            if torch.any(stds <= 0):
                raise ValueError("Statistic normalization std must be positive")
        else:
            means = torch.zeros(self.model.num_stats, dtype=torch.float32)
            stds = torch.ones(self.model.num_stats, dtype=torch.float32)
        self.register_buffer("stat_mean", means.view(1, 1, -1, 1))
        self.register_buffer("stat_std", stds.view(1, 1, -1, 1))

    def _segment_mask(self, batch_size, device, step=None):
        count = int(round(self.args.mask_ratio * self.model.need_token))
        count = max(1, min(self.model.need_token - 1, count))
        step = int(self.global_step if step is None else step)
        generator = torch.Generator(device=device)
        generator.manual_seed(self.args.seed + 1000003 * step)
        scores = torch.rand(
            batch_size,
            self.model.need_token,
            device=device,
            generator=generator,
        )
        selected = scores.topk(count, dim=1, largest=False).indices
        mask = torch.zeros(
            batch_size, self.model.need_token, dtype=torch.bool, device=device
        )
        mask.scatter_(1, selected, True)
        return mask

    def _shuffle_offset(self, batch_size, device, step=None):
        if batch_size < 2:
            raise ValueError("target shuffling requires batch_size >= 2")
        step = int(self.global_step if step is None else step)
        generator = torch.Generator(device=device)
        generator.manual_seed(self.args.seed + 1000003 * step + 7919)
        return int(
            torch.randint(
                1,
                batch_size,
                (1,),
                device=device,
                generator=generator,
            ).item()
        )

    def _masked_encoder_input(self, batch, mask):
        segmented = self.model.segment_input(batch).clone()
        segmented.masked_fill_(mask[:, None, :, None], 0.0)
        return segmented.reshape_as(batch)
        
    def _legacy_training_step(self, batch, batch_idx):

        # store the checkpoint every 5000 steps
        # Implementation note.
        if self.global_step % 2000 == 0:
            self.trainer.save_checkpoint(
                filepath=f"{self.save_path}/epoch={self.current_epoch}_step={self.global_step}.ckpt"
            )
        # Implementation note.
        # prest_samples, shhs_samples = batch
        # Implementation note.
        chbmit_samples, tuev_samples = batch  
        # Implementation note.
        # print("batch_chbmit_samples:",len(chbmit_samples))
        # print("batch_tuev_samples:",len(tuev_samples))
        # print(seed_samples)
        # print(seed_samples[0].shape)  #0
        # print("batch_seed_samples:",len(seed_samples))
        

        contrastive_loss = 0

        # if len(prest_samples) > 0:
        #     """
        #     For prest
        #     """
        # Implementation note.
        #     prest_masked_emb, prest_samples_emb = self.model(prest_samples, 0)  #x,n_channel_offset=0

        # Implementation note.
        # Implementation note.
        #     prest_masked_emb = F.normalize(prest_masked_emb, dim=1, p=2)
        # Implementation note.

        #     # representation similarity matrix, NxN
        # Implementation note.
        # Implementation note.
        # Implementation note.
        # Implementation note.
        # Implementation note.
        # Implementation note.
        # Implementation note.

        # """
        # For shhs
        # """
        # Implementation note.

        # # For shhs
        # Implementation note.
        # shhs_masked_emb = F.normalize(shhs_masked_emb, dim=1, p=2)
        # N = shhs_samples_emb.shape[0]

        # Implementation note.
        # logits = torch.mm(shhs_samples_emb, shhs_masked_emb.t()) / self.T
        # labels = torch.arange(N).to(logits.device)
        # contrastive_loss += F.cross_entropy(logits, labels, reduction="mean")
        if len(chbmit_samples) > 0:
            # Implementation note.
            chbmit_raw_emb, chbmit_pred_emb = self.model(chbmit_samples)  
            
            # Implementation note.
            # Implementation note.
            pred_flat = chbmit_pred_emb.view(chbmit_pred_emb.shape[0], -1)  # shape: (batch, 16*20)
            raw_flat = chbmit_raw_emb.view(chbmit_raw_emb.shape[0], -1)      # shape: (batch, 16*20)
            
            # Implementation note.
            euclidean_dist = torch.norm(pred_flat - raw_flat, p=2, dim=1)  # shape: (batch,)
            
            # Implementation note.
            contrastive_loss += euclidean_dist.mean()

        if len(tuev_samples) > 0:
            """
            For TUEV (flag=1)
            """
            # Implementation note.
            tuev_raw_emb, tuev_pred_emb = self.model(tuev_samples)  
            
            # Implementation note.
            # Implementation note.
            pred_flat = tuev_pred_emb.view(tuev_pred_emb.shape[0], -1)  # shape: (batch, 16*20)
            raw_flat = tuev_raw_emb.view(tuev_raw_emb.shape[0], -1)      # shape: (batch, 16*20)
            
            # Implementation note.
            euclidean_dist = torch.norm(pred_flat - raw_flat, p=2, dim=1)  # shape: (batch,)
            
            # Implementation note.
            contrastive_loss += euclidean_dist.mean()

        # if len(seed_samples) > 0:
        #     """
        #     For SEED (flag=2)
        #     """
        # Implementation note.
        #     seed_raw_emb, seed_pred_emb = self.model(seed_samples)  
            
        # Implementation note.
        # Implementation note.
        #     pred_flat = seed_pred_emb.view(seed_pred_emb.shape[0], -1)  # shape: (batch, 16*20)
        #     raw_flat = seed_raw_emb.view(seed_raw_emb.shape[0], -1)      # shape: (batch, 16*20)
            
        # Implementation note.
        #     euclidean_dist = torch.norm(pred_flat - raw_flat, p=2, dim=1)  # shape: (batch,)
            
        # Implementation note.
        #     contrastive_loss += euclidean_dist.mean()

        self.log("train_loss", contrastive_loss)  # Implementation note.
        return contrastive_loss

    def training_step(self, batch, batch_idx):
        objective = self.args.objective
        if objective == "unmasked_stats":
            target, prediction = self.model(batch)
            per_statistic_loss = (prediction - target).square().mean(
                dim=(0, 1, 3)
            )
            loss = per_statistic_loss.mean()
        else:
            mask = self._segment_mask(batch.shape[0], batch.device)
            encoder_input = self._masked_encoder_input(batch, mask)
            target, prediction = self.model(batch, encoder_input)
            if objective in (
                "normalized_masked_stats",
                "shuffled_masked_stats",
            ):
                target = (target - self.stat_mean) / self.stat_std
                if objective == "shuffled_masked_stats":
                    if target.shape[0] < 2:
                        raise ValueError(
                            "shuffled_masked_stats requires batch_size >= 2"
                        )
                    offset = self._shuffle_offset(
                        target.shape[0], target.device
                    )
                    target = torch.roll(target, shifts=offset, dims=0)
                squared = (prediction - target).square()
                expanded_mask = mask[:, None, None, :].expand_as(squared)
                per_statistic_loss = torch.stack(
                    [
                        squared[:, :, index, :][
                            expanded_mask[:, :, index, :]
                        ].mean()
                        for index in range(self.model.num_stats)
                    ]
                )
                loss = per_statistic_loss.mean()
            else:
                squared = (prediction - target).square().mean(dim=-1)
                expanded_mask = mask[:, None, :].expand_as(squared)
                loss = squared[expanded_mask].mean()
                per_statistic_loss = None

        if per_statistic_loss is not None:
            for index, stat_name in enumerate(self.model.stats_features):
                self.log(
                    f"train_loss_{stat_name}",
                    per_statistic_loss[index],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss


    # Implementation note.
    # Implementation note.
    # def configure_optimizers(self):
    # Implementation note.
    #     optimizer = torch.optim.Adam(
    #         self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
    #     )

    # Implementation note.
    #     scheduler = torch.optim.lr_scheduler.StepLR(
    #         optimizer, step_size=10000, gamma=0.3
    #     )

    #     return [optimizer], [scheduler]
    def configure_optimizers(self):  # Implementation note.
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        return [optimizer]  # , [scheduler]

# from utils import TUEVCHBMITLoader,collate_fn_tuevchbmit_datasets  #TUAB TUEVCHBMITLoader collate_fn_tuevchbmit_datasets
# from utils import TUABCHBMITLoader,collate_fn_tuabchbmit_datasets #TUEV  TUABCHBMITLoader collate_fn_tuabchbmit_datasets
# from utils import TUABTUEVLoader,collate_fn_tuabtuev_datasets #CHBMIT TUABTUEVLoader collate_fn_tuabtuev_datasets

def source_inventory(loader):
    roots = {
        "tuab": loader.base.root_tuab,
        "tuev": loader.base.root_tuev,
        "chbmit": loader.base.root_chbmit,
    }
    paths = {
        "tuab": loader.base.tuab_list,
        "tuev": loader.base.tuev_list,
        "chbmit": loader.base.chbmit_list,
    }
    digest = hashlib.sha256()
    counts = {}
    for dataset in loader.datasets:
        relative = sorted(
            os.path.relpath(path, roots[dataset]).replace(os.sep, "/")
            for path in paths[dataset]
        )
        counts[dataset] = len(relative)
        for path in relative:
            digest.update(dataset.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest(), counts


def prepare_dataloader(args):
    # define the (seizure) data loader
    root_tuab = args.tuab_root
    root_tuev = args.tuev_root
    root_chbmit = args.chbmit_root

    '''
Implementation detail.
    '''
    dataset_pairs = {
        "tuevchbmit": ("tuev", "chbmit"),
        "tuabchbmit": ("tuab", "chbmit"),
        "tuabtuev": ("tuab", "tuev"),
    }
    held_out_datasets = {
        "tuevchbmit": "tuab",
        "tuabchbmit": "tuev",
        "tuabtuev": "chbmit",
    }
    if args.logdataset not in dataset_pairs:
        raise ValueError(
            f"Unsupported logdataset={args.logdataset!r}; "
            f"choose from {sorted(dataset_pairs)}"
        )
    args.held_out_dataset = held_out_datasets[args.logdataset]
    loader = TwoDatasetPretrainLoader(
        root_tuab,
        root_tuev,
        root_chbmit,
        dataset_pairs[args.logdataset],
        target_sample_rate=args.sampling_rate,
        target_window_seconds=args.window_seconds,
    )
    if args.stat_normalization:
        normalization = json.loads(
            Path(args.stat_normalization).read_text(encoding="utf-8")
        )
        current_sha, current_counts = source_inventory(loader)
        if current_sha != normalization.get(
            "source_relative_path_inventory_sha256"
        ):
            raise RuntimeError(
                "Pretraining source inventory differs from statistic "
                "normalization artifact"
            )
        if current_counts != normalization.get("base_source_record_counts"):
            raise RuntimeError(
                "Pretraining source counts differ from statistic "
                "normalization artifact"
            )
    

    
    # Implementation note.
    # Implementation note.
    # loader = TUABTUEVLoader(root_tuab, root_tuev)
    # loader = TUABTUEVCHBMITLoader(root_tuab, root_tuev, root_chbmit) 
    sampler = None
    if args.source_sampling == "balanced":
        weights = torch.empty(len(loader), dtype=torch.double)
        offset = 0
        for dataset in loader.datasets:
            length = loader.lengths[dataset]
            weights[offset : offset + length] = 1.0 / length
            offset += length
        sampler = torch.utils.data.WeightedRandomSampler(
            weights,
            num_samples=len(loader),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    train_loader = torch.utils.data.DataLoader(
        loader,  # Implementation note.
        batch_size=args.batch_size,
        shuffle=args.source_sampling == "proportional",
        sampler=sampler,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        drop_last=True,  # Implementation note.
        # Implementation note.
        # Implementation note.
        # '''
        # Implementation note.
        # '''
        collate_fn=collate_fn_two_dataset_pretrain,

        # Implementation note.
        # Implementation note.
        # collate_fn=collate_fn_tuabtuev_datasets,
    )
    
    return train_loader
 
 
def pretrain(args):
    if not 0.0 < args.mask_ratio < 1.0:
        raise ValueError("mask_ratio must be strictly between zero and one")
    if args.batch_size < 2 and args.objective == "shuffled_masked_stats":
        raise ValueError("shuffled_masked_stats requires batch_size >= 2")
    seed_everything(args.seed, args.deterministic)
    pl.seed_everything(args.seed, workers=True)
    # get data loaders
    train_loader = prepare_dataloader(args)

    # Implementation note.
    dataset_name = args.logdataset  # Implementation note.


    # Implementation note.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Implementation note.
    N_version = (
        dataset_name + "_" + timestamp
    )
    # define the model
    save_path = f"log-pretrain/{N_version}-unsupervised/checkpoints"  # Implementation note.
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Implementation note.
    model = LitModel_supervised_pretrain(args, save_path)  # Implementation note.
    
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        version=f"{N_version}/checkpoints",
        name="log-pretrain",
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        every_n_train_steps=args.checkpoint_every_n_steps,
        save_top_k=-1,
        save_last=True,
        filename="{epoch:03d}-{step:08d}",
    )
    trainer = pl.Trainer(
        devices=[0],  # Implementation note.
        accelerator="gpu",
        strategy="auto",
        auto_select_gpus=False,  # Implementation note.
        benchmark=not args.deterministic,
        deterministic="warn" if args.deterministic else False,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        limit_train_batches=args.limit_train_batches,
        callbacks=[checkpoint_callback],
    )

    # train the model
    trainer.fit(model, train_loader)
    metrics = {
        key: value
        for key, value in trainer.callback_metrics.items()
        if isinstance(value, (int, float, torch.Tensor))
    }
    result_path = save_experiment_result(
        args,
        metrics,
        extra={
            "selected_checkpoint": artifact_metadata(
                checkpoint_callback.last_model_path
            ),
            "selection_rule": f"fixed final optimizer step {args.max_steps}",
            "held_out_dataset": args.held_out_dataset,
            "stat_normalization": (
                artifact_metadata(args.stat_normalization)
                if args.stat_normalization
                else None
            ),
            "mask_segments": (
                int(round(args.mask_ratio * args.need_token))
                if args.objective != "unmasked_stats"
                else 0
            ),
            "mask_schedule": (
                "independent device generator seeded as "
                "seed + 1000003 * global_step; identical across masked objectives"
                if args.objective != "unmasked_stats"
                else None
            ),
            "shuffle_schedule": (
                "independent device generator seeded as "
                "seed + 1000003 * global_step + 7919"
                if args.objective == "shuffled_masked_stats"
                else None
            ),
        },
    )
    print(f"result_json: {result_path}")

#python run_unsupervised_pretrain.py --logdataset tuabchbmit --need_token 20
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("--sampling_rate", type=int, default=200)
    parser.add_argument("--window_seconds", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=12, help="number of workers")
    parser.add_argument("--need_token", type=int, default=20, help="number of pharse divided")
    parser.add_argument("--feature_out", type=int, default=299)
    parser.add_argument(
        "--objective",
        choices=UnsupervisedPretrain.SUPPORTED_OBJECTIVES,
        default="unmasked_stats",
    )
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument(
        "--stat_normalization",
        type=str,
        default="",
        help="source-training-only normalization JSON for masked statistic objectives",
    )
    parser.add_argument(
        "--source_sampling",
        choices=["balanced", "proportional"],
        default="balanced",
    )

    parser.add_argument("--logdataset", type=str, default="tuevchbmit", help="which datasets to use")
    parser.add_argument("--gate_type", choices=["uniform", "static", "input"], default="uniform")
    parser.add_argument("--gate_hidden_dim", type=int, default=16)
    parser.add_argument("--expert_axis", choices=["legacy", "temporal"], default="temporal")
    parser.add_argument(
        "--norm_type",
        choices=["auto", "legacy_global", "local_filter"],
        default="local_filter",
    )
    parser.add_argument(
        "--gate_descriptor_source",
        choices=["raw", "encoded"],
        default="raw",
    )
    parser.add_argument("--match_dynamic_budget", action="store_true")
    parser.add_argument(
        "--expert_kernels",
        type=str,
        default="7",
        help="comma-separated odd temporal kernels",
    )
    parser.add_argument(
        "--relation_mode",
        choices=[
            "none",
            "attention",
            "static",
            "dynamic",
            "dynamic_normalized",
            "spatial1x1",
            "static_conditioned_matched",
            "spatial1x1_conditioned_matched",
            "dynamic_normalized_attention",
        ],
        default="none",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--result_dir", type=str, default="results/pretrain")
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--log_dir", type=str, default=".")
    parser.add_argument(
        "--tuab_root",
        type=str,
        default="/home/dataset/tuh_eeg/tuh_eeg_abnormal/tuh_eeg_abnormal/v3.0.1/edf/processed",
    )
    parser.add_argument(
        "--tuev_root",
        type=str,
        default="/home/dataset/tuh_eeg/tuh_eeg_events/tuh_eeg_events/v2.0.1/edf",
    )
    parser.add_argument(
        "--chbmit_root",
        type=str,
        default="/home/dataset/CHB-MIT/clean_segments",
    )
    args = parser.parse_args()
    print(args)

    pretrain(args)
    
    
