"""Evaluate a checkpoint on the validation split, or predict specific patients.

  python test.py --config config.yaml --checkpoint outputs/run_X/checkpoints/best.pt
  python test.py --config config.yaml --checkpoint best.pt --patient path/to/BraTS2021_00495
"""
import argparse
import os
from datetime import datetime

import torch

from config import get_device, load_config, seed_everything
from data_loader import list_patients, split_patients
from display import plot_per_patient
from evaluate import evaluate_patients, per_patient_table, predict_volume, save_nifti, summarize
from metrics import dice_iou
from model import build_model
from train import make_run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--patient", nargs="*", help="patient folder(s) to predict; default = val split")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = get_device(cfg["device"])
    run_dir = make_run_dir(cfg, args.config, prefix="test")

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[setup] device {device} | checkpoint epoch {ckpt['epoch']} | output {run_dir}")

    # ---- Inference mode: predict given patients and save NIfTI (seg file not required)
    if args.patient:
        from preprocessing import load_patient
        for p in args.patient:
            pid = os.path.basename(os.path.normpath(p))
            image, seg, affine = load_patient(p, cfg, need_seg=False)
            pred, _ = predict_volume(model, image, cfg, device, debug=args.debug)
            out = os.path.join(run_dir, "predictions", f"{pid}_pred.nii.gz")
            save_nifti(pred, affine, out)
            msg = f"[predict] {pid}: {int(pred.sum())} tumor voxels -> {out}"
            if seg is not None:
                d, j = dice_iou(pred, seg > 0)
                msg += f" | dice {d:.4f} iou {j:.4f}"
            print(msg)
        return

    # ---- Evaluation mode: same seed + val_fraction as training -> same val patients
    patients = list_patients(cfg["paths"]["data_root"], cfg["data"]["max_patients"])
    _, val_patients = split_patients(patients, cfg["data"]["val_fraction"], cfg["seed"])
    rows, _ = evaluate_patients(model, val_patients, cfg, device,
                                pred_dir=os.path.join(run_dir, "predictions"),
                                overlay_dir=os.path.join(run_dir, "graphs", "overlays"),
                                debug=args.debug)
    s = summarize(rows)
    plot_per_patient(rows, os.path.join(run_dir, "graphs", "per_patient_dice.png"))

    report = (f"Test results  {datetime.now():%Y-%m-%d %H:%M:%S}\n"
              f"checkpoint {os.path.abspath(args.checkpoint)} (epoch {ckpt['epoch']})\n"
              f"device {device}, val patients {len(val_patients)}\n"
              f"3D Dice {s['dice_mean']:.4f} ± {s['dice_std']:.4f} | IoU {s['iou_mean']:.4f} ± {s['iou_std']:.4f}\n"
              f"false-positive slices {s['fp_slices']}\n\n{per_patient_table(rows)}\n")
    with open(os.path.join(run_dir, "results.txt"), "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
