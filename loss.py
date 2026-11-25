"""
All loss terms used by train.py.

  - classification_loss: main proficiency-classification loss on the fused
    ego+exo feature (CORN ordinal loss if cfg.loss == "ordinal", else CE)
  - infonce_loss:         InfoNCE contrastive loss between ego and exo query
                           tokens (enabled via cfg.CVML)
  - gradient_blending_loss: Gradient-Blending style loss (enabled via
                           cfg.VIBGB). Ego-only and exo-only branches are
                           each run through a VIB bottleneck and scored on a
                           validation batch vs. a training batch; the
                           "overfitting gap" of each branch sets its loss
                           weight (branches that overfit faster get
                           down-weighted), plus the VIB KL regularizer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from coral_pytorch.dataset import proba_to_label
from coral_pytorch.losses import corn_loss

_ce = nn.CrossEntropyLoss()


def classification_loss(cfg, logit, label):
    """Returns (loss, predicted_labels) for the main classification term."""
    if cfg.loss == "ordinal":
        loss = corn_loss(logit, label, num_classes=cfg.num_classes)
        predicted_labels = proba_to_label(logit).float()
    else:
        loss = _ce(logit, label)
        predicted_labels = logit.argmax(dim=1)
    return loss, predicted_labels


def infonce_loss(features_ego, features_exo, temperature=0.05):
    """Symmetric InfoNCE loss pulling together ego/exo tokens from the same
    (time-step, sample) pair, pushing apart all other pairs in the batch."""
    ego_tokens = F.normalize(features_ego, dim=-1).transpose(0, 1).reshape(-1, features_ego.shape[-1])
    exo_tokens = F.normalize(features_exo, dim=-1).transpose(0, 1).reshape(-1, features_exo.shape[-1])
    logits = torch.matmul(ego_tokens, exo_tokens.T) / temperature
    labels = torch.arange(ego_tokens.size(0)).to(ego_tokens.device)
    return F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)


def gradient_blending_loss(cfg, neck, head, ego_vib, exo_vib, ego_data, exo_list, label, onehot,
                            val_ego_data, val_exo_list, val_label, val_onehot):
    """Gradient-Blending (VIBGB) loss for one training step.

    Runs the ego-only and exo-only branches (other modality zeroed) through
    the neck + a VIB bottleneck, on both the current train batch and a
    sampled val batch. The gap between train and val loss estimates how
    much each branch is overfitting; branches that overfit more get a
    smaller weight `w_*` in the combined loss, and each branch also
    contributes its VIB KL divergence.
    """
    # --- val-batch losses, no grad, used only to measure the train/val gap ---
    neck.eval(); head.eval(); ego_vib.eval(); exo_vib.eval()
    with torch.no_grad():
        fusion_val_ego, *_ = neck(val_ego_data, [torch.zeros_like(e) for e in val_exo_list], val_onehot, train=False)
        fusion_val_exo, *_ = neck(torch.zeros_like(val_ego_data), val_exo_list, val_onehot, train=False)

        fusion_val_ego, _ = ego_vib(fusion_val_ego)
        fusion_val_exo, _ = exo_vib(fusion_val_exo)

        loss_ego_val = corn_loss(head(fusion_val_ego.mean(1).detach()), val_label, num_classes=cfg.num_classes)
        loss_exo_val = corn_loss(head(fusion_val_exo.mean(1).detach()), val_label, num_classes=cfg.num_classes)
    neck.train(); head.train(); ego_vib.train(); exo_vib.train()

    # --- train-batch losses (with grad) for the same ego-only / exo-only branches ---
    fusion_train_ego, *_ = neck(ego_data, [torch.zeros_like(e) for e in exo_list], onehot, train=True)
    fusion_train_exo, *_ = neck(torch.zeros_like(ego_data), exo_list, onehot, train=True)

    kl_ego, kl_exo = 0.0, 0.0
    if cfg.VIB:
        fusion_train_ego, kl_ego = ego_vib(fusion_train_ego)
        fusion_train_exo, kl_exo = exo_vib(fusion_train_exo)

    loss_ego_train = corn_loss(head(fusion_train_ego.mean(1)), label, num_classes=cfg.num_classes)
    loss_exo_train = corn_loss(head(fusion_train_exo.mean(1)), label, num_classes=cfg.num_classes)

    # overfitting-to-generalization gap per branch (clamped at 0: a branch
    # that generalizes *better* than it trains gets no penalty)
    gap_ego = torch.clamp(loss_ego_val - loss_ego_train, min=0.)
    gap_exo = torch.clamp(loss_exo_val - loss_exo_train, min=0.)

    eps = 1e-6
    w_ego = 1 / (gap_ego + eps)
    w_exo = 1 / (gap_exo + eps)
    w_sum = w_ego + w_exo
    w_ego, w_exo = w_ego / w_sum, w_exo / w_sum

    gb_loss = w_ego * loss_ego_train + w_exo * loss_exo_train
    kl_loss = kl_ego + kl_exo if cfg.VIB else 0.0
    return gb_loss, kl_loss
