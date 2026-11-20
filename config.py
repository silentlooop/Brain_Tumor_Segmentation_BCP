import os
import random

import numpy as np
import torch
import yaml


def load_config(path):
    """Read config.yaml and make relative paths absolute (relative to the yaml file)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    base = os.path.dirname(os.path.abspath(path))
    for key in ("data_root", "output_dir"):
        p = os.path.expanduser(cfg["paths"][key])
        cfg["paths"][key] = p if os.path.isabs(p) else os.path.join(base, p)

    validate(cfg)
    return cfg


def validate(cfg):
    if cfg["task"] != "binary":
        # MULTICLASS extension point. To add it:
        #   1. preprocessing.make_target already remaps label 4 -> 3 and returns class indices
        #   2. loss_function.build_loss already has a CrossEntropy + multi-class Dice branch
        #   3. still TODO: set model.out_channels: 4, use softmax/argmax in evaluate.predict_volume,
        #      and compute Dice/IoU per class in metrics.py
        raise NotImplementedError("Only task: binary is wired end to end. See config.validate().")

    step = 2 ** cfg["model"]["depth"]
    if cfg["data"]["crop_size"] % step:
        raise ValueError(f"crop_size must be divisible by 2**depth = {step}")
    if cfg["model"]["in_channels"] != len(cfg["data"]["modalities"]):
        raise ValueError("model.in_channels must equal the number of modalities")
    if cfg["slice_selection"]["gray_zone_policy"] not in ("tumor", "drop"):
        raise ValueError("gray_zone_policy must be 'tumor' or 'drop'")


def get_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
