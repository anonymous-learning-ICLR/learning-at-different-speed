# Learning at Different Speed — TAER-MoE Demo

This repository provides a runnable demonstration of TAER-MoE, including the model and training code, a packaged checkpoint, a small tropical cyclone test sample, and data visualization code with example outputs. You can run the demo to inspect the data, try the evaluation pipeline, and check that the included components work together.

**This is a sample release. We plan to release the full dataset in a subsequent update.** The current package includes only the demonstration test data; training and validation data are not included.

## Included files

All runnable components are in `TAER_MoE_minimal/`:

| File or directory | Purpose |
| --- | --- |
| `train.py` | Single training entry; epoch-based training with optional input acceleration and progress logging |
| `paper_pipeline.py`, `taer_feedback.py` | Training objective, expert-task statistics, and routing feedback |
| `evaluate.py` | Single evaluation entry; automatically detects the new or bundled checkpoint format |
| `visualize.py` | Generate the four data visualization outputs |
| `TCNM/` | Model architectures, data loaders, losses, and utilities |
| `install.sh` | Python dependency installation |
| `seed3047ckpt.pt` | Packaged checkpoint, stored with Git LFS |
| `data_demo/` | MERANTI (2010), WP test example and its required data assets |
| `visualizations/` | One CSV table and three PNG figures |

## Installation

The demo was checked with **Python 3.8.5**, PyTorch 1.11.0, NumPy 1.24.3, OpenCV headless 4.12.0.88, and Matplotlib 3.7.5 on Linux. Use Python 3.8 for the pinned dependencies in `install.sh`.

Install Git and Git LFS, then download the repository and checkpoint:

```bash
git lfs install
git clone https://github.com/anonymous-learning-ICLR/learning-at-different-speed.git
cd learning-at-different-speed
git lfs pull
```

Create an environment and install the Python dependencies:

```bash
conda create -n taer-demo python=3.8 -y
conda activate taer-demo
cd TAER_MoE_minimal
bash install.sh
```

The checkpoint should be approximately **276 MiB**. If it is only a small text file, run `git lfs pull` to download the actual weights.

All commands below run from `TAER_MoE_minimal/`.

## Run the evaluation demo

No training is required. The included checkpoint can be evaluated on CPU:

```bash
python evaluate.py \
  --checkpoint seed3047ckpt.pt \
  --data-dir data_demo \
  --areas WP \
  --device cpu \
  --threads 4 \
  --workers 0 \
  --output demo_results
```

With a compatible CUDA installation, replace `--device cpu` with `--device cuda --gpu_num 0`.

The script prints only the result tables during a successful evaluation and writes the same tables to `demo_results/summary.txt`. Each table contains overall ADE and FDE, per-step TDE, pressure and wind errors, and an AVG row. Track errors are in km, pressure errors in hPa, and wind errors in m/s.

The demo contains **one evaluation window**: eight observed records followed by four future records from MERANTI in the WP basin. Therefore, its `Area: ALL` and `Area: WP` tables are identical. A CPU check with the bundled defaults produced:

| Demo metric | Value |
| --- | ---: |
| Overall ADE (km) | 29.394611 |
| Overall FDE (km) | 37.714573 |
| Average pressure error (hPa) | 7.7907 |
| Average wind error (m/s) | 5.2079 |

These numbers describe the bundled example, not the full test dataset. Small numerical differences can occur across environments.

The checkpoint contains multiple model parameter sets and their saved fusion coefficients in a single file. Evaluation uses the saved settings, including ground-truth Best-of-K candidate selection. The historical fusion coefficients were fitted on test data; these demo results should not be interpreted as an independent held-out benchmark or as forecasts made without access to future labels.

## Visualize the example

Visualization runs on CPU and does not require loading the checkpoint:

```bash
python visualize.py --data-dir data_demo --output visualizations
```

This writes exactly four files:

| Output | Contents |
| --- | --- |
| [data_1d.csv](TAER_MoE_minimal/visualizations/data_1d.csv) | Readable table of UTC timestamps, observed/future roles, longitude, latitude, pressure, and wind |
| [environment_3d.png](TAER_MoE_minimal/visualizations/environment_3d.png) | Stacked GPH500 maps over observation time |
| [features6.png](TAER_MoE_minimal/visualizations/features6.png) | Six-channel feature maps |
| [features10.png](TAER_MoE_minimal/visualizations/features10.png) | Ten-channel feature maps |

The environmental figure uses **time × grid y × grid x** at a single pressure level; its third axis is time. Feature figures display the stored map values. Open the CSV in a spreadsheet application to inspect the 1D data.

### Example figures

![GPH500 environment over observation time](TAER_MoE_minimal/visualizations/environment_3d.png)

![Six-channel feature maps](TAER_MoE_minimal/visualizations/features6.png)

![Ten-channel feature maps](TAER_MoE_minimal/visualizations/features10.png)

## Training code and completeness checks

`train.py` is the only training entry, and `evaluate.py` is the only evaluation entry. The model, data loader, losses, expert-task statistics, and feedback controller are included. Inspect its options with:

```bash
python train.py --help
```

Training requires separately authorized training and validation data. The bundled test example is intended for evaluation and visualization and is insufficient for training. `--data-root` is the parent of the following folders:

```text
TropiCycloneNet1/
  bst_divi10_train_val_test_inlcude15_2023_new/<AREA>/{train,val,test}/*.txt
  all_area_correct_location_includeunder15_2023/
  all_ocean_gph500_2023/
  features6/
  features10/
```

Preserve the environmental assets' basin/year/storm/timestamp organization. Run on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --data-root /path/to/TropiCycloneNet1 \
  --output-dir ./runs/full_seed3407_tc3_inputs \
  --variant full \
  --seed 3407 \
  --device cuda \
  --areas EP NA NI SI SP WP \
  --epochs 100 \
  --batch-size 16 \
  --workers 0 \
  --threads 2 \
  --feature-mode legacy \
  --feature-reduction none \
  --split-policy existing
```

Keep the explicit seed and batch size in this command, since the parser's generic defaults differ. Each epoch traverses the full training loader, including the final partial batch. After each epoch, the controller collects training and validation expert-task errors, computes routing corrections, and applies them through `set_feedback()` before evaluating and saving the updated state.

`--feature-mode legacy --feature-reduction none` retains the original Feature 6/10 handling: `timestamp.npy` lookup, original storm-name fallback, zero/partial-load fallback, and no spatial pooling. Files with `_features6.npy` or `_features10.npy` suffixes are ignored by this mode. The visualization code displays the stored maps independently of this training lookup.

The run directory contains `last.pt` after each completed epoch, `best.pt` when validation ADE improves, `feedback.jsonl`, the configuration, and the split manifest. Resume in the same output directory by adding `--resume ./runs/full_seed3407_tc3_inputs/last.pt`. Source hashes and the split manifest are checked on resume. A narrowly scoped compatibility check also accepts checkpoints from the original epoch driver before its entry files were consolidated. This driver saves at epoch boundaries.

For optional input acceleration and progress logging, keep the same `train.py` command and append:

```text
--io-workers 1 --prefetch-factor 2 --cache-frames 4096 --pin-memory --log-every 50
```

This adds ordered input prefetching, caching, pinned transfers, and `OUTPUT_DIR/out.log` without changing the training loop or its configuration. Use a fresh output directory or explicitly resume an existing run.

Evaluate a newly trained checkpoint with:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --checkpoint ./runs/full_seed3407_tc3_inputs/best.pt \
  --data-root /path/to/TropiCycloneNet1 \
  --device cuda \
  --output ./runs/full_seed3407_tc3_inputs/test_metrics.json
```

`evaluate.py` detects the new `taer-paper-reconstruction-v1` format and keeps its saved routing feedback fixed. It prints and saves aggregate metrics as JSON. By default, it requires train/validation/test folders to audit split overlap. `--demo-only` skips that audit for test-only demonstrations but still requires test data for the selected areas. By default these are all areas configured in the checkpoint; `--areas WP` can restrict a demo to WP when that basin is in the saved configuration.

The previously supplied `seed3047ckpt.pt` retains its original format and uses the same `evaluate.py` command, as shown above. The evaluator selects the appropriate architecture and loading path automatically; the checkpoint formats themselves remain distinct. For the bundled checkpoint, `--output` names a directory containing `summary.txt`; for a new epoch-model checkpoint, it names a JSON file. Only the necessary legacy architecture defaults are retained in the evaluator; there is no separate legacy training entry. No private training/validation data or newly trained checkpoint is included in this update.

Run the implementation checks from `TAER_MoE_minimal/`:

```bash
python -m unittest discover -s tests -v
```

These cover routing feedback, expert-task statistics, data handling, model gradients, saving/resuming training, evaluation, and input-pipeline consistency. Full training requires the separately authorized data.

## Planned data release

We plan to make the full dataset available in a subsequent release and update this README with download links, data preparation instructions, and commands for full-dataset experiments.

