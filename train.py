import argparse
import csv
import os
import shutil
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import get_device, load_config, seed_everything
from data_loader import (TrainSliceStream, count_buckets, list_patients, scan_split,
                         select_train_slices, split_patients)
from display import plot_buckets, plot_dice_iou, plot_loss, plot_per_patient
from evaluate import evaluate_patients, per_patient_table, summarize
from loss_function import build_loss
from metrics import batch_slice_dice
from model import build_model


def make_run_dir(cfg, config_path, prefix="run"):
    run_dir = os.path.join(cfg["paths"]["output_dir"], f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}")
    for sub in ("checkpoints", "predictions", "graphs/overlays"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    shutil.copy(config_path, os.path.join(run_dir, "config_used.yaml"))  # exact copy of the file
    return run_dir


def overfit_one_batch(model, loader, criterion, optimizer, cfg, device, run_dir):
    """Sanity check: a working pipeline must be able to memorise one batch (loss -> ~0).
    If it can't, something is broken (labels, loss, shapes), so there is no point training."""
    images, masks = next((b for b in loader if b[1].sum() > 0), next(iter(loader)))
    images, masks = images.to(device), masks.to(device)
    print(f"[sanity] overfitting one batch {tuple(images.shape)} for {cfg['sanity']['overfit_steps']} steps")

    model.train()
    losses = []
    for step in range(1, cfg["sanity"]["overfit_steps"] + 1):
        loss = criterion(model(images), masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 20 == 0 or step == 1:
            print(f"[sanity] step {step:4d} loss {loss.item():.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(losses), plt.yscale("log"), plt.xlabel("Step"), plt.ylabel("Loss"), plt.title("Overfit one batch")
    plt.tight_layout(), plt.savefig(os.path.join(run_dir, "graphs", "overfit_loss.png"), dpi=120), plt.close()
    verdict = "PASS" if losses[-1] < 0.1 else "CHECK: loss did not get close to 0"
    with open(os.path.join(run_dir, "results.txt"), "w") as f:
        f.write(f"Overfit-one-batch sanity check\nfirst loss {losses[0]:.4f}\n"
                f"final loss {losses[-1]:.4f}\n{verdict}\n")
    print(f"[sanity] first {losses[0]:.4f} -> final {losses[-1]:.4f}  {verdict}")


def train_one_epoch(model, loader, criterion, optimizer, scaler, cfg, device, debug):
    model.train()
    use_amp = scaler.is_enabled()
    loss_sum, dice_sum, n = 0.0, 0.0, 0
    for i, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        if debug and i == 0:
            print(f"[train] images {tuple(images.shape)} masks {tuple(masks.shape)} "
                  f"mask values {masks.unique().tolist()}")

        with torch.autocast(device.type, enabled=use_amp):
            logits = model(images, debug=debug and i == 0)
            loss = criterion(logits.float(), masks)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item() * len(images)
        dice_sum += batch_slice_dice(logits.detach().float(), masks, cfg["training"]["threshold"]) * len(images)
        n += len(images)
    return loss_sum / n, dice_sum / n


def write_results(path, info):
    s_best, s_final = info["best_summary"], info["final_summary"]
    fmt = lambda s: (f"Dice {s['dice_mean']:.4f} ± {s['dice_std']:.4f} | "
                     f"IoU {s['iou_mean']:.4f} ± {s['iou_std']:.4f}")
    cfg = info["cfg"]
    lines = [
        "Brain Tumor Segmentation (2D U-Net, BraTS 2021) - training results",
        "=" * 66,
        f"Date/time        : {info['start']:%Y-%m-%d %H:%M:%S}",
        f"Device           : {info['device']}",
        f"Training time    : {info['train_time'] / 60:.1f} min",
        "",
        "Dataset",
        f"  data_root      : {cfg['paths']['data_root']}",
        f"  patients       : train {info['n_train']}, val {info['n_val']}",
        f"  train slices used for training: {info['n_train_slices']}",
    ]
    for split, counts in info["buckets"].items():
        lines.append(f"  {split:<5} buckets  : " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    t, m = cfg["training"], cfg["model"]
    lines += [
        "",
        "Key config",
        f"  task {cfg['task']}, crop {cfg['data']['crop_size']}, modalities {cfg['data']['modalities']}",
        f"  model base_filters {m['base_filters']}, depth {m['depth']}",
        f"  epochs {t['epochs']}, batch {t['batch_size']}, lr {t['lr']}, weight_decay {t['weight_decay']}",
        f"  loss {t['bce_weight']}*BCE + {t['dice_weight']}*Dice, pos_weight {t['pos_weight']}, "
        f"threshold {t['threshold']}",
        f"  slice selection {cfg['slice_selection']}",
        "",
        "Results (3D, per patient, all slices)",
        f"  best epoch     : {info['best_epoch']}",
        f"  best  val      : {fmt(s_best)}   (best.pt)",
        f"  final val      : {fmt(info['last_summary'])}   (last epoch)",
        f"  false-positive slices (best.pt): {s_final['fp_slices']}",
        "",
        "Per-patient (best.pt)",
        per_patient_table(info["rows"]),
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--debug", action="store_true", help="print tensor shapes")
    args = parser.parse_args()

    start = datetime.now()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = get_device(cfg["device"])
    run_dir = make_run_dir(cfg, args.config)
    print(f"[setup] device {device} | run dir {run_dir}")

    # ---- Data: split by patient, then bucket every slice
    patients = list_patients(cfg["paths"]["data_root"], cfg["data"]["max_patients"])
    train_patients, val_patients = split_patients(patients, cfg["data"]["val_fraction"], cfg["seed"])
    print(f"[data] {len(patients)} patients -> train {len(train_patients)}, val {len(val_patients)}")

    train_buckets = scan_split(train_patients, cfg)
    val_buckets = scan_split(val_patients, cfg)
    buckets = {"train": count_buckets(train_buckets), "val": count_buckets(val_buckets)}
    for split, counts in buckets.items():
        print(f"[data] {split} slice buckets: {counts}")
    plot_buckets(buckets, os.path.join(run_dir, "graphs", "slice_buckets.png"))

    # Slice selection is for the TRAINING split only. Validation must look like real use:
    # a new scan arrives and the model has to handle every slice, including empty ones.
    rng = np.random.default_rng(cfg["seed"])
    train_slices = [(p, select_train_slices(train_buckets[p], cfg["slice_selection"], rng))
                    for p in train_patients]
    dataset = TrainSliceStream(train_slices, cfg, augment=True)
    print(f"[data] training on {len(dataset)} selected slices")

    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"],
                        num_workers=cfg["training"]["num_workers"])

    # ---- Model, loss, optimizer
    model = build_model(cfg).to(device)
    criterion = build_loss(cfg).to(device)  # .to() moves pos_weight too
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                                  weight_decay=cfg["training"]["weight_decay"])
    use_amp = cfg["training"]["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"[setup] {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, "
          f"mixed precision {use_amp}")

    if cfg["sanity"]["overfit_one_batch"]:
        dataset.augment = False
        overfit_one_batch(model, loader, criterion, optimizer, cfg, device, run_dir)
        return

    # ---- Training loop
    history, best_dice, best_epoch = [], -1.0, 0
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    t0 = time.time()
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        dataset.set_epoch(epoch)
        tl, td = train_one_epoch(model, loader, criterion, optimizer, scaler, cfg, device, args.debug and epoch == 1)
        rows, vl = evaluate_patients(model, val_patients, cfg, device, criterion, debug=args.debug and epoch == 1)
        s = last_summary = summarize(rows)

        history.append({"epoch": epoch, "train_loss": tl, "train_dice": td, "val_loss": vl,
                        "val_dice": s["dice_mean"], "val_iou": s["iou_mean"],
                        "lr": optimizer.param_groups[0]["lr"]})
        print(f"[epoch {epoch:3d}] train loss {tl:.4f} dice {td:.4f} | val loss {vl:.4f} "
              f"3D dice {s['dice_mean']:.4f} iou {s['iou_mean']:.4f} fp-slices {s['fp_slices']}")

        with open(os.path.join(run_dir, "history.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=history[0].keys())
            w.writeheader()
            w.writerows(history)

        ckpt = {"model": model.state_dict(), "epoch": epoch, "val_dice": s["dice_mean"], "config": cfg}
        torch.save(ckpt, os.path.join(ckpt_dir, "last.pt"))
        if s["dice_mean"] > best_dice:
            best_dice, best_epoch, best_summary = s["dice_mean"], epoch, s
            torch.save(ckpt, os.path.join(ckpt_dir, "best.pt"))
            print(f"[epoch {epoch:3d}] new best val dice {best_dice:.4f} -> best.pt")
    train_time = time.time() - t0

    # ---- Final evaluation with the best checkpoint: predictions, overlays, report
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best.pt"), map_location=device)["model"])
    rows, _ = evaluate_patients(model, val_patients, cfg, device,
                                pred_dir=os.path.join(run_dir, "predictions"),
                                overlay_dir=os.path.join(run_dir, "graphs", "overlays"))
    graphs = os.path.join(run_dir, "graphs")
    plot_loss(history, os.path.join(graphs, "loss_curve.png"))
    plot_dice_iou(history, os.path.join(graphs, "dice_iou_curve.png"))
    plot_per_patient(rows, os.path.join(graphs, "per_patient_dice.png"))

    write_results(os.path.join(run_dir, "results.txt"), {
        "cfg": cfg, "start": start, "device": device, "train_time": train_time,
        "n_train": len(train_patients), "n_val": len(val_patients), "n_train_slices": len(dataset),
        "buckets": buckets, "best_epoch": best_epoch, "best_summary": best_summary,
        "last_summary": last_summary, "final_summary": summarize(rows), "rows": rows,
    })
    print(f"[done] best epoch {best_epoch}, val 3D dice {best_dice:.4f}. Results in {run_dir}")


if __name__ == "__main__":
    main()
