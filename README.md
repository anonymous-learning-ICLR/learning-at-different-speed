# TAER-MoE

Training, testing, and tropical cyclone visualization examples.
`data_test` contains the selected Nadine (1974) test case (4 windows); `data_demo` contains the Meranti (2010) visualization example. The full dataset will be released later.

## Install

Install Git LFS, then run:

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

```bash
python test.py --checkpoint ckpt.pt --data-dir data_test --device cpu
```

For GPU 0, use `CUDA_VISIBLE_DEVICES=0 python test.py --checkpoint ckpt.pt --data-dir data_test --device cuda`.
The script prints ALL and regional error tables and saves `test_report/summary.txt`.

## Visualize

```bash
python visualize.py --data-dir data_demo --output visualizations
```

Outputs: [1D table](TAER_MoE_minimal/visualizations/data_1d.csv), [3D environment](TAER_MoE_minimal/visualizations/environment_3d.png), [Feature 6](TAER_MoE_minimal/visualizations/features6.png), and [Feature 10](TAER_MoE_minimal/visualizations/features10.png).

![3D environment](TAER_MoE_minimal/visualizations/environment_3d.png)
![Feature 6](TAER_MoE_minimal/visualizations/features6.png)
![Feature 10](TAER_MoE_minimal/visualizations/features10.png)

## Train

Training requires separately obtained training and validation data in the same dataset layout.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data-root /path/to/TropiCycloneNet1 --output-dir runs/seed200 \
  --variant full --seed 200 --device cuda --areas EP NA NI SI SP WP \
  --epochs 101 --batch-size 64 --workers 4 --threads 4 \
  --feature-mode legacy --feature-reduction none --split-policy existing
```
