# Learning at Different Speed — TAER-MoE Demo

A runnable sample with training code, a checkpoint, one tropical cyclone test case, and visualization examples. Try the commands below to inspect the data and results.

**This is a demo release. The full dataset will be released later.** Training and validation data are not included.

## Install

Use Python 3.8 and install Git LFS first.

```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/anonymous-learning-ICLR/learning-at-different-speed.git
cd learning-at-different-speed
git lfs pull --include="TAER_MoE_minimal/ckpt.pt"
conda create -n taer-demo python=3.8 -y
conda activate taer-demo
cd TAER_MoE_minimal
bash install.sh
```

## Test

Run from `TAER_MoE_minimal/`:

```bash
python test.py --checkpoint ckpt.pt --data-dir data_demo --areas WP --device cpu --threads 4 --workers 0 --output demo_results
```

For GPU 0, replace `--device cpu` with `--device cuda --gpu_num 0`.

The script prints only ALL and regional error tables, also saved to `demo_results/summary.txt`. The bundled example contains one WP test window.

The supplied checkpoint uses test-label calibration and true-future candidate selection. These demo scores are not independent held-out results.

## Visualize

```bash
python visualize.py --data-dir data_demo --output visualizations
```

Four example outputs:

- [1D data table](TAER_MoE_minimal/visualizations/data_1d.csv)
- [3D environment plot](TAER_MoE_minimal/visualizations/environment_3d.png)
- [Feature 6](TAER_MoE_minimal/visualizations/features6.png)
- [Feature 10](TAER_MoE_minimal/visualizations/features10.png)

## Train

Training requires separately obtained training and validation data. See the available options:

```bash
python train.py --help
```
