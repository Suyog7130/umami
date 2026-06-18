#!/usr/bin/env bash
set -eo pipefail

EPOCHS="${1:-10}"

# Useful for logs.
export PYTHONUNBUFFERED=1

export PATH="/home/suyog.garg/.conda/envs/phd/bin:/usr/local/bin:/usr/bin:/bin"
PYTHON_EXE="/home/suyog.garg/.conda/envs/phd/bin/python"

# Optional debugging. Uncomment only when diagnosing crashes.
# export CUDA_LAUNCH_BLOCKING=1
# export OMP_DISPLAY_ENV=TRUE

# Activate your environment.
# Modify this path/name if needed.
# source "${HOME}/.bashrc"
# conda activate phd
PYTHON_EXE="/home/suyog.garg/.conda/envs/phd/bin/python3"

echo "Checking important environment variables:"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "MKL_NUM_THREADS=${MKL_NUM_THREADS}"
echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
echo "TORCH_NUM_THREADS=${TORCH_NUM_THREADS}"

mkdir -p logs

echo "Checking Python executable..."
ls -l "${PYTHON_EXE}"
"${PYTHON_EXE}" --version

# Activate your specific environment
conda activate phd

# Run your script with arguments
python3 optimize.py --model-config modelconfig-running.json --epochs "${EPOCHS}" --verbose


EXIT_CODE=$?

echo "============================================================"
echo "Finished optimization job with EPOCHS=${EPOCHS}."
echo "Exit code: ${EXIT_CODE}"
