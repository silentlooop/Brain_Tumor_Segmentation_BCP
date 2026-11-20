import numpy as np
import torch


def dice_iou(pred, gt):
    """Dice and IoU for two boolean arrays of any shape (2D slice or 3D volume).

    If both are empty the model correctly said "no tumor", so that is a perfect 1.0
    (the formula alone would give 0/0).
    """
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0, 1.0
    union = total - inter
    return 2 * inter / total, inter / union


def batch_slice_dice(logits, targets, threshold):
    """Mean slice-level Dice for a training batch (used for the per-epoch train curve)."""
    pred = (torch.sigmoid(logits) > threshold).float()
    inter = (pred * targets).sum((1, 2, 3))
    total = pred.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    dice = torch.where(total == 0, torch.ones_like(total), 2 * inter / total.clamp(min=1))
    return dice.mean().item()


def false_positive_slices(pred, gt):
    """Number of axial slices where the model predicts tumor but the ground truth has none.
    pred, gt: (H, W, D)."""
    pred_any = pred.reshape(-1, pred.shape[-1]).any(0)
    gt_any = gt.reshape(-1, gt.shape[-1]).any(0)
    return int(np.sum(pred_any & ~gt_any))
