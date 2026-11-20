import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice on probabilities. `smooth` avoids 0/0 on slices with no tumor at all."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Computed over the whole batch, not per slice: on a slice with no tumor, per-slice
        # Dice is 1 / (sum of probs + 1), and summed over 36k pixels even tiny probabilities
        # keep that near 0, so the loss stays high and noisy. Batch-level Dice avoids that.
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum()
        dice = (2 * inter + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """bce_weight * BCE + dice_weight * Dice.

    Why both: BCE judges every pixel on its own and gives smooth, stable gradients, but
    with ~1% tumor pixels it is happy predicting "no tumor" everywhere. Dice measures
    overlap of the tumor region, so it ignores the huge background, but alone it is noisy.

    Logits vs probabilities: the model outputs raw logits. BCEWithLogitsLoss applies the
    sigmoid internally in a numerically safe way (no log(0)). Dice needs real probabilities
    in [0, 1], so it applies sigmoid itself.
    """

    def __init__(self, bce_weight, dice_weight, pos_weight=None):
        super().__init__()
        pw = torch.tensor([pos_weight]) if pos_weight else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.dice = DiceLoss()
        self.bce_weight, self.dice_weight = bce_weight, dice_weight

    def forward(self, logits, targets):
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


class CEDiceLoss(nn.Module):
    """MULTICLASS extension: CrossEntropy on logits + mean soft Dice over the tumor classes.
    Expects logits (B, K, H, W) and integer targets (B, H, W) with labels 0..K-1."""

    def __init__(self, ce_weight, dice_weight, smooth=1.0):
        super().__init__()
        self.ce_weight, self.dice_weight, self.smooth = ce_weight, dice_weight, smooth

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets)
        probs = logits.softmax(1)
        onehot = F.one_hot(targets, logits.shape[1]).permute(0, 3, 1, 2).float()
        inter = (probs * onehot).sum((0, 2, 3))
        dice = (2 * inter + self.smooth) / (probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3)) + self.smooth)
        return self.ce_weight * ce + self.dice_weight * (1 - dice[1:].mean())  # skip background


def build_loss(cfg):
    t = cfg["training"]
    if cfg["task"] == "binary":
        return BCEDiceLoss(t["bce_weight"], t["dice_weight"], t["pos_weight"])
    return CEDiceLoss(t["bce_weight"], t["dice_weight"])
