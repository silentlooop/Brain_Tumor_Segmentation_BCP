import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

GREEN = (0, 200, 0)  # OpenCV uses BGR, not RGB
RED = (0, 0, 255)


# ---------------------------------------------------------------- OpenCV overlays

def upright(slice_2d):
    """NIfTI slices are stored [x, y]; rot90 turns them so the brain faces up.
    rot90 returns a strided view, and OpenCV needs a contiguous array."""
    return np.ascontiguousarray(np.rot90(slice_2d))


def to_bgr(gray):
    """z-scored float slice -> 8-bit BGR image (1st-99th percentile of the brain -> 0..255)."""
    brain = gray[gray != 0]
    lo, hi = np.percentile(brain, [1, 99]) if brain.size else (0, 1)
    img = np.clip((gray - lo) / (hi - lo + 1e-8), 0, 1) * 255
    img[gray == 0] = 0
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def draw_overlay(image, mask, color, label, alpha=0.4):
    """Blend `mask` onto a BGR uint8 `image` in `color`, box it, and write `label: area px`."""
    mask = mask.astype(bool)
    colored = image.copy()
    colored[mask] = color
    out = cv2.addWeighted(colored, alpha, image, 1 - alpha, 0)
    out[~mask] = image[~mask]  # only tint the mask, keep the rest at full brightness

    # One box per connected blob, so scattered false positives don't create one giant box.
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 1)

    if contours:
        x, y, _, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))  # label the largest blob
        org, text = (x, max(y - 5, 12)), f"{label}: {int(mask.sum())} px"
    else:
        org, text = (4, 15), f"{label}: none"
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def overlay_panel(flair, gt, pred, title, scale=2):
    """FLAIR | ground truth (green) | prediction (red) for one axial slice, as one BGR image."""
    flair, gt, pred = upright(flair), upright(gt).astype(np.uint8), upright(pred).astype(np.uint8)
    size = (flair.shape[1] * scale, flair.shape[0] * scale)
    base = cv2.resize(to_bgr(flair), size, interpolation=cv2.INTER_LINEAR)
    # Masks are upscaled with NEAREST so they stay hard 0/1 labels.
    gt = cv2.resize(gt, size, interpolation=cv2.INTER_NEAREST)
    pred = cv2.resize(pred, size, interpolation=cv2.INTER_NEAREST)

    panels = [base.copy(), draw_overlay(base, gt, GREEN, "GT"), draw_overlay(base, pred, RED, "Pred")]
    for p, name in zip(panels, ["FLAIR", "Ground truth", "Prediction"]):
        cv2.putText(p, name, (4, p.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    header = np.zeros((24, size[0] * 3, 3), np.uint8)
    cv2.putText(header, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([header, cv2.hconcat(panels)])


def save_patient_overlays(flair, gt, pred, pid, out_dir):
    """Saves the slice with the most tumor and one healthy slice. Volumes are (H, W, D)."""
    os.makedirs(out_dir, exist_ok=True)
    tumor_z = int(gt.sum((0, 1)).argmax())
    brain = (flair != 0).sum((0, 1))
    healthy = np.where((gt.sum((0, 1)) == 0) & (brain > 0))[0]
    picks = [("tumor", tumor_z)]
    if len(healthy):
        picks.append(("healthy", int(healthy[brain[healthy].argmax()])))  # healthy slice with most brain
    for kind, z in picks:
        img = overlay_panel(flair[..., z], gt[..., z], pred[..., z], f"{pid}  z={z}  ({kind})")
        cv2.imwrite(os.path.join(out_dir, f"{pid}_{kind}_z{z}.png"), img)


# ---------------------------------------------------------------- matplotlib graphs

def plot_loss(history, path):
    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    plt.plot(epochs, [h["val_loss"] for h in history], marker="o", label="val")
    plt.xlabel("Epoch"), plt.ylabel("BCE + Dice loss"), plt.title("Loss"), plt.legend(), plt.grid(alpha=0.3)
    plt.tight_layout(), plt.savefig(path, dpi=120), plt.close()


def plot_dice_iou(history, path):
    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [h["val_dice"] for h in history], marker="o", label="val 3D Dice")
    plt.plot(epochs, [h["val_iou"] for h in history], marker="o", label="val 3D IoU")
    plt.plot(epochs, [h["train_dice"] for h in history], ls="--", label="train slice Dice")
    plt.ylim(0, 1), plt.xlabel("Epoch"), plt.title("Dice / IoU"), plt.legend(), plt.grid(alpha=0.3)
    plt.tight_layout(), plt.savefig(path, dpi=120), plt.close()


def plot_per_patient(rows, path):
    rows = sorted(rows, key=lambda r: r["dice"])
    plt.figure(figsize=(max(6, 0.3 * len(rows)), 4))
    plt.bar(range(len(rows)), [r["dice"] for r in rows], color="#2E86AB")
    plt.xticks(range(len(rows)), [r["patient"].replace("BraTS2021_", "") for r in rows], rotation=90, fontsize=7)
    plt.ylim(0, 1), plt.ylabel("3D Dice"), plt.title("Per-patient validation Dice")
    plt.tight_layout(), plt.savefig(path, dpi=120), plt.close()


def plot_buckets(counts_by_split, path):
    names = list(next(iter(counts_by_split.values())).keys())
    x = np.arange(len(names))
    width = 0.8 / len(counts_by_split)
    plt.figure(figsize=(7, 4))
    for i, (split, counts) in enumerate(counts_by_split.items()):
        bars = plt.bar(x + i * width, [counts[n] for n in names], width, label=split)
        plt.bar_label(bars, fontsize=7)
    plt.xticks(x + width * (len(counts_by_split) - 1) / 2, names)
    plt.ylabel("Slices"), plt.title("Slice buckets"), plt.legend()
    plt.tight_layout(), plt.savefig(path, dpi=120), plt.close()


# ---------------------------------------------------------------- quick look (notebooks)

def show_patient(patient_dir):
    """Matplotlib view of all 4 modalities at the slice with the most tumor."""
    import nibabel as nib
    pid = os.path.basename(os.path.normpath(patient_dir))
    load = lambda name: nib.load(os.path.join(patient_dir, f"{pid}_{name}.nii.gz")).get_fdata()

    seg = load("seg")
    z = int(np.argmax(seg.sum(axis=(0, 1))))
    mask = upright(seg[:, :, z])

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, m in zip(axes, ["flair", "t1", "t1ce", "t2"]):
        ax.imshow(upright(load(m)[:, :, z]), cmap="gray")
        ax.imshow(np.ma.masked_where(mask == 0, mask), cmap="Reds", alpha=0.45)
        ax.set_title(m.upper())
        ax.axis("off")
    fig.suptitle(f"{pid} | slice z = {z}")
    plt.tight_layout()
    plt.show()
