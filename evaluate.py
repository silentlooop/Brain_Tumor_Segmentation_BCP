import os

import nibabel as nib
import numpy as np
import torch

from display import save_patient_overlays
from metrics import dice_iou, false_positive_slices
from preprocessing import crop_hw, load_patient, make_target, uncrop_hw


@torch.no_grad()
def predict_volume(model, image, cfg, device, target=None, criterion=None, debug=False):
    """Predict every axial slice of one patient.

    image: (C, H, W, D) normalized. Returns a binary (H, W, D) uint8 mask at the ORIGINAL
    size, plus the summed loss over slices if target/criterion are given.
    """
    size = cfg["data"]["crop_size"]
    _, H, W, D = image.shape
    x = torch.from_numpy(crop_hw(image, size, 1, 2)).permute(3, 0, 1, 2)  # (D, C, h, w)
    y = None
    if target is not None:
        y = torch.from_numpy(crop_hw(target, size, 0, 1)).permute(2, 0, 1)[:, None]  # (D, 1, h, w)

    use_amp = cfg["training"]["mixed_precision"] and device.type == "cuda"
    probs, loss_sum = [], 0.0
    bs = cfg["evaluation"]["batch_size"]
    for s in range(0, D, bs):
        xb = x[s:s + bs].to(device)
        with torch.autocast(device.type, enabled=use_amp):
            logits = model(xb)
        if criterion is not None:
            loss_sum += criterion(logits.float(), y[s:s + bs].to(device)).item() * len(xb)
        probs.append(torch.sigmoid(logits.float()).cpu())
        if debug and s == 0:
            print(f"[eval] slices {tuple(xb.shape)} -> logits {tuple(logits.shape)}")

    probs = torch.cat(probs)[:, 0].permute(1, 2, 0).numpy()  # (h, w, D)
    pred = (probs > cfg["training"]["threshold"]).astype(np.uint8)
    return uncrop_hw(pred, H, W, 0, 1), loss_sum


def save_nifti(mask, affine, path):
    """Saving with the ORIGINAL affine puts the prediction in the same physical space as the
    input scan, so it overlays correctly in viewers like ITK-SNAP or 3D Slicer."""
    img = nib.Nifti1Image(mask.astype(np.uint8), affine)
    img.set_data_dtype(np.uint8)
    nib.save(img, path)


def evaluate_patients(model, patients, cfg, device, criterion=None,
                      pred_dir=None, overlay_dir=None, debug=False):
    """3D Dice / IoU per patient on ALL slices (no slice filtering: that is what the model
    will face on a new scan). Optionally saves NIfTI predictions and overlay images."""
    model.eval()
    rows, loss_sum, n_slices = [], 0.0, 0
    for i, patient_dir in enumerate(patients):
        pid = os.path.basename(os.path.normpath(patient_dir))
        image, seg, affine = load_patient(patient_dir, cfg)
        target = make_target(seg, cfg["task"])
        pred, loss = predict_volume(model, image, cfg, device, target, criterion, debug and i == 0)

        gt = seg > 0
        dice, iou = dice_iou(pred, gt)
        rows.append({"patient": pid, "dice": float(dice), "iou": float(iou),
                     "fp_slices": false_positive_slices(pred, gt),
                     "gt_voxels": int(gt.sum()), "pred_voxels": int(pred.sum())})
        loss_sum += loss
        n_slices += image.shape[-1]

        if pred_dir:
            os.makedirs(pred_dir, exist_ok=True)
            save_nifti(pred, affine, os.path.join(pred_dir, f"{pid}_pred.nii.gz"))
        if overlay_dir and i < cfg["evaluation"]["num_overlay_patients"]:
            flair = image[cfg["data"]["modalities"].index("flair")]
            save_patient_overlays(flair, gt, pred, pid, overlay_dir)

    return rows, (loss_sum / n_slices if criterion else float("nan"))


def summarize(rows):
    d = np.array([r["dice"] for r in rows])
    j = np.array([r["iou"] for r in rows])
    # plain Python floats: numpy scalars can't be loaded back with torch.load(weights_only=True)
    return {"dice_mean": float(d.mean()), "dice_std": float(d.std()),
            "iou_mean": float(j.mean()), "iou_std": float(j.std()),
            "fp_slices": int(sum(r["fp_slices"] for r in rows))}


def per_patient_table(rows):
    lines = [f"{'patient':<20}{'dice':>8}{'iou':>8}{'fp_slices':>11}{'gt_vox':>10}{'pred_vox':>10}"]
    for r in rows:
        lines.append(f"{r['patient']:<20}{r['dice']:>8.4f}{r['iou']:>8.4f}{r['fp_slices']:>11}"
                     f"{r['gt_voxels']:>10}{r['pred_voxels']:>10}")
    return "\n".join(lines)
