#!/bin/sh
#SBATCH --job-name=image_steganography_with_ECC    # Job name
#SBATCH --ntasks=1                                 # Run on a single CPU
#SBATCH --time=24:00:00                            # Time limit hrs:min:sec
#SBATCH --output=test_mm_%j.out                    # Standard output and error log
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --partition=dgx

CUDA_HOME=/usr/local/cuda
CUDA_VISIBLE_DEVICES=1
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

nvcc --version
nvidia-smi
python -u train.py
