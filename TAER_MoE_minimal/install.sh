#!/usr/bin/env bash
# Tested with Python 3.8. Install an appropriate CUDA PyTorch build for GPU use.
set -euo pipefail
python -m pip install numpy==1.24.3 torch==1.11.0 opencv-python-headless==4.12.0.88 matplotlib==3.7.5 scipy==1.10.1 scikit-learn==1.3.2
