"""
Train + evaluate AdaMVS (redundancy-aware ego-exo fusion for proficiency
estimation). There is no separate test-only script: each epoch runs a full
pass over `train` then a full pass over `test` and reports test accuracy.

Usage:
    python train.py --config configs/egoexo4d.py

The loss is composed of up to 3 terms, each toggled by the config; see
loss.py for their implementation:
  - cls_loss:  main classification loss on the fused ego+exo feature
  - VIBGB:     Gradient-Blending style loss (ego-only vs exo-only branches)
  - CVML:      InfoNCE contrastive loss between ego and exo query tokens
"""
import os
import json
import wandb
import torch
from torch.optim import lr_scheduler
from itertools import cycle
from tqdm import tqdm

from utils import parse_args, init_seed, init_gpu, get_logger
from dataloader import get_dataloader
from models import build_neck, build_head
from models.head import VIBLayer
from loss import classification_loss, infonce_loss, gradient_blending_loss

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")  # overridden by cfg.multi_gpu in init_gpu()


def unpack_batch(cfg, data, device, zero_ego=False, zero_exo=False):
    """Unpack one dataloader batch into (ego, exo_list, label, onehot) on `device`.

    `zero_ego`/`zero_exo` replace the corresponding branch's input with
    zeros, used by the VIBGB ego-only / exo-only forward passes below.
    """
    if cfg.dataset == "egoexofitness":
        ego_data, exo_data, exo2_data, exo3_data, label, name, onehot = data
        exo_list = [exo_data, exo2_data, exo3_data]
    else:  # egoexo4d: 4 exocentric views
        ego_data, exo_data, exo2_data, exo3_data, exo4_data, label, name, onehot, file_name = data
        exo_list = [exo_data, exo2_data, exo3_data, exo4_data]

    ego_data = ego_data.float().to(device)
    exo_list = [e.float().to(device) for e in exo_list]
    label = label.to(device)
    onehot = onehot.long().to(device)

    if zero_ego:
        ego_data = torch.zeros_like(ego_data)
    if zero_exo:
        exo_list = [torch.zeros_like(e) for e in exo_list]

    return ego_data, exo_list, label, onehot


def main():
    cfg = parse_args()
    init_seed(cfg)
    multi_gpu, device = init_gpu(cfg)

    data_loaders = get_dataloader(cfg)

    neck = build_neck(cfg)
    head = build_head(cfg)

    # VIB bottlenecks used only by the Gradient-Blending (VIBGB) loss below
    ego_vib = VIBLayer(cfg.head_in_dim, cfg.vib_dim).cuda()
    exo_vib = VIBLayer(cfg.head_in_dim, cfg.vib_dim).cuda()

    optimizer = torch.optim.AdamW([
        {'params': neck.parameters(), 'lr': cfg.lr},
        {'params': ego_vib.parameters(), 'lr': cfg.lr},
        {'params': exo_vib.parameters(), 'lr': cfg.lr},
        {'params': head.parameters(), 'lr': cfg.lr}],
        lr=cfg.lr,
        weight_decay=1e-2)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler_step, gamma=0.1)

    os.makedirs("exp", exist_ok=True)
    base_logger = get_logger(f'exp/{cfg.logname}.log', "EXP1")
    base_logger.info("Start training...")
    base_logger.info(f"Config:\n{cfg.pretty_text}")
    wandb.init(mode="disabled")  # switch to "online" (and set up your wandb project) to log remotely

    best_val_acc = 0.0  # best test accuracy over epochs (what the paper reports)
    gb_val_source = cfg.get("gb_val_source", "test")
    eval_splits = ["val", "test"] if "val" in data_loaders else ["test"]
    best_holdout_acc, test_at_best_holdout = -1.0, None
    history = []

    for epoch in range(cfg.epoch_num):
        epoch_acc = {}
        gb_stats = []
        for split in ["train"] + eval_splits:
            if split == "train":
                head.train(); neck.train(); ego_vib.train(); exo_vib.train()
                torch.set_grad_enabled(True)
            else:
                head.eval(); neck.eval()
                torch.set_grad_enabled(False)

            total_correct, total_loss, n_batches = 0, 0.0, 0
            # cycles over the val source so a val batch is always available
            # for the VIBGB gap loss during training
            val_iter = cycle(data_loaders["val" if gb_val_source == "holdout" else "test"])

            for data_ in tqdm(data_loaders[split], desc=f"epoch {epoch} [{split}]"):
                ego_data, exo_list, label, onehot = unpack_batch(cfg, data_, device)

                # --- redundancy-aware ego/exo fusion + classification ---
                fusion_feat, features_ego, features_exo, _, _, _, egoexoscore, exo_loss_ = neck(
                    ego_data, exo_list, onehot, train=(split == "train"), topk=cfg.topk, tau=cfg.tau)
                logit = head(fusion_feat.mean(1))

                cls_loss, predicted_labels = classification_loss(cfg, logit, label)
                loss = cls_loss
                if cfg.CVML:
                    loss = loss + cfg.cvml_rate * infonce_loss(features_ego, features_exo)

                if cfg.VIBGB and split == "train":
                    val_data = next(val_iter)
                    val_ego, val_exo_list, val_label, val_onehot = unpack_batch(cfg, val_data, device)
                    gb_loss, kl_loss, stats = gradient_blending_loss(
                        cfg, neck, head, ego_vib, exo_vib,
                        ego_data, exo_list, label, onehot,
                        val_ego, val_exo_list, val_label, val_onehot)
                    gb_stats.append(stats)
                    loss = loss + cfg.get("gb_rate", 1.0) * gb_loss + cfg.vib_rate * kl_loss

                total_correct += (predicted_labels == label).sum().item()
                total_loss += loss.item()
                n_batches += 1

                if split == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            acc = total_correct / len(data_loaders[split].dataset)
            avg_loss = total_loss / max(n_batches, 1)
            if split == "test" and acc > best_val_acc:
                best_val_acc = acc
                print("New Best Found !!")

            msg = f"[Epoch {epoch}] {split} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Best Acc: {best_val_acc:.4f}"
            print(msg)
            base_logger.info(msg)

            epoch_acc[split] = acc
            if split == "train":
                scheduler.step()

        if gb_stats:
            mean = lambda k: sum(s[k] for s in gb_stats) / len(gb_stats)
            msg = (f"[Epoch {epoch}] GB | w_ego: {mean('w_ego'):.3f} | gap_ego: {mean('gap_ego'):.4f}"
                   f" | gap_exo: {mean('gap_exo'):.4f} | both_gaps_zero: {mean('both_gaps_zero'):.3f}")
            print(msg)
            base_logger.info(msg)
        if "val" in epoch_acc and epoch_acc["val"] > best_holdout_acc:
            best_holdout_acc, test_at_best_holdout = epoch_acc["val"], epoch_acc["test"]
        history.append(epoch_acc)

    base_logger.info("Training End")
    base_logger.info(f"Best Acc: {best_val_acc:.4f}")
    summary = {"best_test_acc": best_val_acc, "last_test_acc": history[-1]["test"],
               "best_holdout_acc": best_holdout_acc if "val" in data_loaders else None,
               "test_at_best_holdout": test_at_best_holdout}
    base_logger.info("SUMMARY " + json.dumps(summary))


if __name__ == '__main__':
    main()
