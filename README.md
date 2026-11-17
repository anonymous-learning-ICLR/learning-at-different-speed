# TAER-MoE

Tropical cyclone implementation of TAER-MoE, with training code, a pretrained single-model checkpoint, a test example, and a separate visualization example.

## Release contents

| Path | Contents |
| --- | --- |
| `TAER_MoE_minimal/train.py` | Training entry point and feedback update schedule |
| `TAER_MoE_minimal/taer_feedback.py` | Validation-based task weighting and expert-task gap corrections |
| `TAER_MoE_minimal/paper_pipeline.py` | Model construction, losses, statistics, and dataset utilities |
| `TAER_MoE_minimal/training_checkpoints.py` | Checkpoint saving and training-state restoration |
| `TAER_MoE_minimal/checkpoint_compat.py` | Source-version compatibility checks |
| `TAER_MoE_minimal/TCNM/` | Network components and data loader |
| `TAER_MoE_minimal/test.py` | Single-model evaluation |
| `TAER_MoE_minimal/ckpt.pt` | Pretrained model and saved feedback state |
| `TAER_MoE_minimal/data_test/` | Nadine test example with four forecast windows |
| `TAER_MoE_minimal/data_demo/` | Meranti visualization data |
| `TAER_MoE_minimal/visualize.py` | Data visualization script |

The included examples support inspection and a quick run. Full training and validation data are not included. The full dataset will be released later.

## Installation

Install Git LFS, then download the repository and checkpoint:

```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/anonymous-learning-ICLR/learning-at-different-speed.git
cd learning-at-different-speed
git lfs pull --include="TAER_MoE_minimal/ckpt.pt"

conda create -n taer-moe python=3.8 -y
conda activate taer-moe
cd TAER_MoE_minimal
bash install.sh
```

All commands below run from `TAER_MoE_minimal/`. For GPU execution, use a PyTorch installation with CUDA support appropriate for your GPU. CPU execution is sufficient for the bundled test example.

## Test the pretrained checkpoint

```bash
python test.py \
  --checkpoint ckpt.pt \
  --data-dir data_test \
  --device cpu \
  --threads 4 \
  --workers 0 \
  --output test_report
```

For GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --checkpoint ckpt.pt --data-dir data_test --device cuda --output test_report
```

The script prints aggregate and regional error tables and saves `test_report/summary.txt`. It reports track ADE and FDE in km, pressure error in hPa, and wind-speed error in m/s. The model uses its saved routing corrections during inference. Future targets are used only to calculate evaluation metrics.

The bundled example contains four windows from one storm. Its scores describe that example, not the full benchmark. To evaluate another compatible dataset, set `--data-dir` to its root and optionally select basins with `--areas EP NA NI SI SP WP`.

## Visualize the example data

```bash
python visualize.py --data-dir data_demo --output visualizations
```

This generates four files:

- `visualizations/data_1d.csv` — tabulated storm data.
- `visualizations/environment_3d.png` — environmental-field visualization.
- `visualizations/features6.png` — six-channel feature visualization.
- `visualizations/features10.png` — ten-channel feature visualization.

These outputs visualize the supplied data. The visualization command does not require the checkpoint.

## Train

Provide training and validation data in the layout expected by `TCNM/data/paper_data.py`. Preserve the directory names and filename capitalization used by the examples. The data root contains the track directory `bst_divi10_train_val_test_inlcude15_2023_new`, environmental data in `all_area_correct_location_includeunder15_2023`, and fields in `all_ocean_gph500_2023`. Track files are organized by basin and split. Feature inputs use `features6` and `features10` when available, according to the selected feature mode.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data-root /path/to/TropiCycloneNet1 \
  --output-dir runs/full_seed200 \
  --variant full \
  --seed 200 \
  --device cuda \
  --areas EP NA NI SI SP WP \
  --epochs 101 \
  --batch-size 64 \
  --workers 4 \
  --threads 4 \
  --feature-mode legacy \
  --feature-reduction none \
  --split-policy existing
```

Use a distinct output directory for each run. Available variants are:

| Variant | Task weighting | Routing correction |
| --- | --- | --- |
| `mmoe` | Off | Off |
| `weighting_only` | On | Off |
| `routing_only` | Off | On |
| `full` | On | On |

Run `python train.py --help` for all settings. Add `--preflight-only` to a training command to check data loading and forward/backward execution before a full run.

Training writes `best.pt`, `last.pt`, checkpoint records, and feedback logs to the output directory. Numbered snapshots are controlled by `--checkpoint-every`. To resume, reuse the original training settings and add `--resume /path/to/last.pt`. Keep the original run directory and its matching `best.pt` when resuming. Resumption checks configuration, data manifests, and source compatibility.

Evaluate a newly trained model with:

```bash
python test.py --checkpoint runs/full_seed200/best.pt --data-dir /path/to/TropiCycloneNet1 --device cuda
```

## Feedback implementation

At each epoch-end feedback update, the training loop collects training and validation statistics with the model in evaluation mode and the preceding feedback state fixed. The controller computes task weights from smoothed validation progress. It computes expert-task errors by accumulating routing-probability-weighted individual expert losses, then derives corrections from the validation–training gaps with median/MAD normalization.

The training loop calls `control.update(...)` followed by `control.apply(model)`. This calls `model.set_feedback(...)`. Task weights affect the training objective, and routing penalties enter the task-specific softmax logits. Feedback signals are detached from gradient computation and saved in checkpoints. Interval validation for checkpoint selection does not update the controller.

The current training path uses the feedback controller. The legacy TSI interface is not used for these updates.
