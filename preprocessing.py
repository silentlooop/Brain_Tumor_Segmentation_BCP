import glob
import os

import nibabel as nib
import numpy as np


def find_file(patient_dir, name):
    """BraTS2021_00495/BraTS2021_00495_flair.nii.gz (Kaggle copies are sometimes plain .nii)."""
    pid = os.path.basename(os.path.normpath(patient_dir))
    matches = glob.glob(os.path.join(patient_dir, f"{pid}_{name}.nii*"))
    if not matches:
        raise FileNotFoundError(f"{pid}_{name}.nii[.gz] not found in {patient_dir}")
    return matches[0]


def load_nifti(patient_dir, name):
    img = nib.load(find_file(patient_dir, name))
    return np.asarray(img.dataobj, dtype=np.float32), img.affine


def normalize_volume(vol, cfg_norm):
    """Z-score one modality of one patient, using brain voxels only.

    Why brain only: ~80% of the volume is black background. Including it would drag the
    mean towards 0 and shrink the std, so the brain intensities would land at different
    values for each patient. Background is set back to exactly 0 afterwards so the model
    still sees a clean "no brain here" signal.
    """
    brain = vol > 0
    if not brain.any():
        return np.zeros_like(vol)

    values = vol[brain]
    if cfg_norm["clip"]:
        # A few very bright voxels (vessels, artefacts) would otherwise inflate the std.
        lo, hi = np.percentile(values, [cfg_norm["clip_low"], cfg_norm["clip_high"]])
        vol = np.clip(vol, lo, hi)
        values = vol[brain]

    out = (vol - values.mean()) / (values.std() + 1e-8)
    out[~brain] = 0
    return out.astype(np.float32)


def load_patient(patient_dir, cfg, need_seg=True):
    """Returns image (C, H, W, D) normalized, seg (H, W, D) raw labels (or None), and the affine.

    The affine maps voxel indices to millimetres in scanner space. All BraTS volumes are
    co-registered, so they share one affine. We keep it so predictions can be written back
    into exactly the same physical space and line up with the scan in any viewer.
    """
    channels, affine = [], None
    for m in cfg["data"]["modalities"]:
        vol, affine = load_nifti(patient_dir, m)
        channels.append(normalize_volume(vol, cfg["normalization"]))

    seg = None
    try:
        seg = load_nifti(patient_dir, "seg")[0].astype(np.uint8)
    except FileNotFoundError:
        if need_seg:
            raise
    return np.stack(channels), seg, affine


def make_target(seg, task):
    if task == "binary":
        return (seg > 0).astype(np.float32)  # whole tumor = labels 1 + 2 + 4
    # MULTICLASS: BraTS skips label 3, so remap 4 -> 3 to get contiguous classes 0..3
    target = seg.astype(np.int64)
    target[target == 4] = 3
    return target


def crop_hw(arr, size, h_axis, w_axis):
    """Center-crop or zero-pad axes h_axis and w_axis to (size, size).

    Image and mask always go through this same function, so they stay pixel-aligned.
    Cropping/padding never interpolates, so mask labels are never blurred.
    """
    for axis in (h_axis, w_axis):
        n = arr.shape[axis]
        if n > size:
            start = (n - size) // 2
            arr = np.take(arr, range(start, start + size), axis=axis)
        elif n < size:
            pad = [(0, 0)] * arr.ndim
            before = (size - n) // 2
            pad[axis] = (before, size - n - before)
            arr = np.pad(arr, pad)
    return arr


def uncrop_hw(arr, orig_h, orig_w, h_axis, w_axis):
    """Inverse of crop_hw: put a cropped prediction back into the original (orig_h, orig_w) grid."""
    for axis, orig in ((h_axis, orig_h), (w_axis, orig_w)):
        n = arr.shape[axis]
        if n < orig:  # we cropped -> pad back with background
            pad = [(0, 0)] * arr.ndim
            before = (orig - n) // 2
            pad[axis] = (before, orig - n - before)
            arr = np.pad(arr, pad)
        elif n > orig:  # we padded -> cut the padding off
            start = (n - orig) // 2
            arr = np.take(arr, range(start, start + orig), axis=axis)
    return arr
