import os
from collections import Counter

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from preprocessing import crop_hw, load_nifti, load_patient, make_target

BUCKETS = ["empty", "healthy", "tumor", "gray_zone"]


# ---------------------------------------------------------------- patients and split

def list_patients(data_root, max_patients=None):
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data_root does not exist: {data_root}")
    patients = sorted(
        os.path.join(data_root, d) for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d)) and d.startswith("BraTS")
    )
    if not patients:
        raise FileNotFoundError(f"No BraTS* patient folders in {data_root}")
    return patients[:max_patients] if max_patients else patients


def split_patients(patients, val_fraction, seed):
    """Split by PATIENT, never by slice.

    Neighbouring slices of one brain look almost identical. If slice 80 is in train and
    slice 81 in validation, the model is tested on something it has basically seen and
    the validation score is far too optimistic.
    """
    if len(patients) == 1:
        print("[split] WARNING: only 1 patient, using it for BOTH train and val (smoke test only)")
        return patients, patients
    order = np.random.default_rng(seed).permutation(len(patients))
    n_val = min(max(1, round(val_fraction * len(patients))), len(patients) - 1)
    val = sorted(patients[i] for i in order[:n_val])
    train = sorted(patients[i] for i in order[n_val:])
    return train, val


# ---------------------------------------------------------------- slice buckets

def bucket_slices(brain_frac, tumor_px, cfg_sel):
    """Label every axial slice of one patient as empty / healthy / tumor / gray_zone."""
    buckets = np.full(len(tumor_px), "healthy", dtype=object)
    buckets[brain_frac < cfg_sel["min_brain_fraction"]] = "empty"
    # Gray zone = a few tumor pixels. They are not healthy, so they never go in that bucket.
    buckets[(tumor_px > 0) & (tumor_px < cfg_sel["min_tumor_pixels"])] = "gray_zone"
    buckets[tumor_px >= cfg_sel["min_tumor_pixels"]] = "tumor"
    return buckets


def select_train_slices(buckets, cfg_sel, rng):
    """Pick which slices of one TRAINING patient to train on.

    ~40% of a BraTS volume is empty or nearly empty, and most brain slices have no tumor.
    Training on all of them teaches the model that "predict nothing" is almost always
    right. Keeping every tumor slice plus a fixed ratio of healthy ones balances that,
    while still showing it healthy brain so it learns not to hallucinate tumors.
    """
    tumor = list(np.where(buckets == "tumor")[0])
    if cfg_sel["gray_zone_policy"] == "tumor":
        tumor += list(np.where(buckets == "gray_zone")[0])
    healthy = np.where(buckets == "healthy")[0]
    n_healthy = min(len(healthy), round(cfg_sel["healthy_keep_ratio"] * len(tumor)))
    keep = tumor + list(rng.choice(healthy, n_healthy, replace=False))
    return sorted(int(z) for z in keep)


class _ScanDataset(Dataset):
    """Wraps the per-patient scan so a DataLoader can run it in parallel processes."""

    def __init__(self, patients):
        self.patients = patients

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, i):
        # Only seg + FLAIR are read: BraTS is skull-stripped, so FLAIR > 0 marks the brain.
        flair, _ = load_nifti(self.patients[i], "flair")
        seg, _ = load_nifti(self.patients[i], "seg")
        brain_frac = (flair > 0).mean(axis=(0, 1))
        tumor_px = (seg > 0).sum(axis=(0, 1))
        return i, brain_frac, tumor_px


def scan_split(patients, cfg):
    """Returns {patient_dir: bucket array} for every patient in the split."""
    loader = DataLoader(_ScanDataset(patients), batch_size=None,
                        num_workers=cfg["data"]["scan_workers"])
    out = {}
    for i, brain_frac, tumor_px in loader:
        out[patients[i]] = bucket_slices(np.asarray(brain_frac), np.asarray(tumor_px),
                                         cfg["slice_selection"])
    return out


def count_buckets(bucket_map):
    counts = Counter()
    for b in bucket_map.values():
        counts.update(b.tolist())
    return {k: counts.get(k, 0) for k in BUCKETS}


# ---------------------------------------------------------------- training dataset

class TrainSliceStream(IterableDataset):
    """Streams 2D training slices: (4, crop, crop) image and (1, crop, crop) mask.

    The full dataset does not fit in RAM, and decompressing 5 NIfTI files for every single
    slice would be painfully slow. So we load `shuffle_buffer_patients` patients at once,
    mix all their selected slices, yield them, and move on. Patient order is reshuffled
    every epoch. With num_workers > 0 each worker streams a different subset of patients.
    """

    def __init__(self, patient_slices, cfg, augment=True):
        self.patient_slices = [(p, z) for p, z in patient_slices if z]  # [(patient_dir, [z, ...])]
        self.cfg = cfg
        self.augment = augment
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return sum(len(z) for _, z in self.patient_slices)

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng([self.cfg["seed"], self.epoch, wid])

        order = np.random.default_rng([self.cfg["seed"], self.epoch]).permutation(len(self.patient_slices))
        mine = [self.patient_slices[i] for i in order[wid::nw]]
        k = self.cfg["data"]["shuffle_buffer_patients"]

        for start in range(0, len(mine), k):
            buffer = []
            for patient_dir, zs in mine[start:start + k]:
                image, seg, _ = load_patient(patient_dir, self.cfg)
                target = make_target(seg, self.cfg["task"])
                size = self.cfg["data"]["crop_size"]
                image = crop_hw(image, size, 1, 2)   # (C, H, W, D)
                target = crop_hw(target, size, 0, 1)  # (H, W, D)
                buffer += [(image[..., z], target[..., z]) for z in zs]

            for i in rng.permutation(len(buffer)):
                img, mask = buffer[i]
                img = torch.from_numpy(np.ascontiguousarray(img))
                mask = torch.from_numpy(np.ascontiguousarray(mask))[None]
                if self.augment:
                    img, mask = self._augment(img, mask, rng)
                yield img, mask

    def _augment(self, img, mask, rng):
        aug = self.cfg["augmentation"]
        if rng.random() < aug["hflip_prob"]:
            img, mask = img.flip(-1), mask.flip(-1)
        if rng.random() < aug["vflip_prob"]:
            img, mask = img.flip(-2), mask.flip(-2)
        if aug["rotate_degrees"]:
            angle = float(rng.uniform(-aug["rotate_degrees"], aug["rotate_degrees"]))
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
            # Masks must use NEAREST: bilinear would invent values like 0.37 at the border,
            # which are not valid labels (and for multi-class would mix class ids).
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        return img, mask
