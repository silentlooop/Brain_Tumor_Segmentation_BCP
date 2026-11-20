**Brain Tumor Segmentation from MRI Scans Using Deep Learning**

***Project Overview***

This project trains a 2D U-Net (PyTorch, written from scratch) to segment brain tumors in the
BraTS 2021 MRI dataset. Every axial slice is a 4-channel input (FLAIR, T1, T1ce, T2), and the model
predicts a pixel-wise whole-tumor mask. Predictions are stacked back into 3D volumes and scored
per patient with 3D Dice and IoU.


***Repository structure***

```
.
├── btsp.ipynb             # runner: clones this repo, then shape test → smoke test → train → results
├── config.yaml            # every setting
├── train.py, test.py      # entry points
├── *.py                   # modules (see table below)
├── BTSP_training_data/    # your local BraTS scans (git-ignored, not included)
├── requirements.txt
└── README.md
```

***Code layout***

| File | What it does |
|---|---|
| `config.yaml` | Every setting (paths, split, slice selection, normalization, model, training) |
| `config.py` | Loads and checks the config, picks the device (cuda → mps → cpu), seeds everything |
| `preprocessing.py` | NIfTI loading, brain-only z-score normalization, crop/pad, label → target |
| `data_loader.py` | Patient list and patient-level split, slice buckets, training-slice selection, streaming dataset |
| `model.py` | 2D U-Net (`python model.py` runs the shape test) |
| `loss_function.py` | BCE + soft Dice loss (and a multi-class CE + Dice extension) |
| `metrics.py` | Dice / IoU (empty + empty = 1.0), false-positive slice count |
| `evaluate.py` | Whole-volume prediction, per-patient 3D metrics, NIfTI saving with the original affine |
| `display.py` | OpenCV overlays (`draw_overlay`), matplotlib graphs, `show_patient` quick look |
| `train.py` / `test.py` | Entry points |
| `btsp.ipynb` | Runner notebook: clones this repo into the working directory, finds the data, writes `config_run.yaml` (repo `config.yaml` + overrides) and calls `train.py` / `test.py` |

***Data***

The MRI scans are **not included** in this repository. Download BraTS 2021 (Task 1, training data)
from the [official challenge page](http://braintumorsegmentation.org/) or a public mirror, and put
the patient folders in `BTSP_training_data/` (or point `paths.data_root` in `config.yaml`
elsewhere). The expected layout is:

```
BTSP_training_data/            <- paths.data_root
  BraTS2021_00495/
    BraTS2021_00495_flair.nii.gz   (240, 240, 155), 1 mm, co-registered
    BraTS2021_00495_t1.nii.gz
    BraTS2021_00495_t1ce.nii.gz
    BraTS2021_00495_t2.nii.gz
    BraTS2021_00495_seg.nii.gz     0 background, 1 necrotic core, 2 edema, 4 enhancing
  ...
```

***Configure***

Edit `config.yaml`. Relative paths are resolved from the yaml's folder. Useful knobs:
`data.max_patients` (small number for a quick debug run), `training.epochs`, `training.batch_size`,
`training.num_workers`, `sanity.overfit_one_batch`.

***Run on a Mac***

```bash
source .venv/bin/activate
pip install -r requirements.txt
python model.py                                   # U-Net shape test
python train.py --config config.yaml [--debug]    # --debug prints tensor shapes
python test.py --config config.yaml --checkpoint outputs/run_XXXX/checkpoints/best.pt
python test.py --config config.yaml --checkpoint <best.pt> --patient BTSP_training_data/BraTS2021_00495
```

Set `sanity.overfit_one_batch: true` to check that the model can memorise one batch (loss → ~0).

***Run in a cloud notebook (GPU)***

Upload `btsp.ipynb`, attach a BraTS 2021 dataset, turn on the GPU **and internet access**, and run
all cells. The notebook clones this repository into `/kaggle/working/` (or pulls it again on re-runs),
finds the data (extracting `.tar` archives if needed), then runs the shape test, a smoke test, the
overfit check, training, evaluation and single-patient inference, and zips `/kaggle/working/outputs/`.
Settings come from the repo's `config.yaml`; per-run changes go in `RUN_OVERRIDES` in cell 3.
The notebook always runs the code that is **pushed** to GitHub, so push changes before running it.

***Outputs (`outputs/run_YYYYMMDD_HHMMSS/`)***

| File | Meaning |
|---|---|
| `config_used.yaml` | Exact copy of the config for this run |
| `results.txt` | Date, device, patients per split, slice-bucket counts, key settings, best epoch, best/final val 3D Dice & IoU (mean ± std), per-patient table, false-positive slices, training time |
| `history.csv` | Per epoch: train loss, train slice Dice, val loss, val 3D Dice, val 3D IoU, learning rate |
| `checkpoints/best.pt`, `last.pt` | Best by mean val 3D Dice, and the latest epoch |
| `predictions/*_pred.nii.gz` | Predicted masks (uint8, original affine) for the validation patients |
| `graphs/loss_curve.png`, `dice_iou_curve.png` | Training curves |
| `graphs/per_patient_dice.png` | 3D Dice per validation patient |
| `graphs/slice_buckets.png` | Empty / healthy / tumor / gray-zone slice counts per split |
| `graphs/overlays/` | FLAIR \| ground truth (green) \| prediction (red), with boxes and tumor area |

`test.py` writes the same kind of report to `outputs/test_YYYYMMDD_HHMMSS/`.

***Required Tools*** <br>
[Github Tutorial](https://www.freecodecamp.org/news/git-and-github-for-beginners/) <br>
[Jupyter Notebook Tutorial](https://www.datacamp.com/tutorial/tutorial-jupyter-notebook) <br>
[Google Colab Tutorial](https://www.geeksforgeeks.org/websites-apps/how-to-use-google-colab/)
