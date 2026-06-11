#!/usr/bin/env bash
set -euo pipefail

LABEL="${1:-ml2ml-paper-1}"
NUM_INJECTIONS="${2:-10}"
SAMPLER="${3:-nessai}"
PE_RUN_TYPE="${4:-ml2ml}"
NLIVE="${5:-300}"
THRESHOLD="${6:-0.2}"
NESSAI_NPOOL="${7:-8}"
PYTORCH_THREADS="${8:-1}"

echo "============================================================"
echo "Starting inference job"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "PWD: $(pwd)"
echo "LABEL: ${LABEL}"
echo "NUM_INJECTIONS: ${NUM_INJECTIONS}"
echo "SAMPLER: ${SAMPLER}"
echo "PE_RUN_TYPE: ${PE_RUN_TYPE}"
echo "NLIVE: ${NLIVE}"
echo "THRESHOLD: ${THRESHOLD}"
echo "NESSAI_NPOOL: ${NESSAI_NPOOL}"
echo "PYTORCH_THREADS: ${PYTORCH_THREADS}"
echo "============================================================"

# Safety: avoid CUDA multiprocessing explosion unless you explicitly request GPU jobs.
# This prevents every nessai worker from importing CUDA tensors or loading a CUDA model.
export CUDA_VISIBLE_DEVICES=""

# Avoid hidden OpenMP and BLAS oversubscription.
# For first tests, keep these at 1.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TORCH_NUM_THREADS="${PYTORCH_THREADS}"

# Useful for logs.
export PYTHONUNBUFFERED=1

# Optional debugging. Uncomment only when diagnosing crashes.
# export CUDA_LAUNCH_BLOCKING=1
# export OMP_DISPLAY_ENV=TRUE

# Activate your environment.
# Modify this path/name if needed.
source "${HOME}/.bashrc"
conda activate phd

echo "Python path:"
which python3
python3 --version

echo "Checking important environment variables:"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "MKL_NUM_THREADS=${MKL_NUM_THREADS}"
echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
echo "TORCH_NUM_THREADS=${TORCH_NUM_THREADS}"

mkdir -p logs

python3 inference.py \
    --label "${LABEL}" \
    --num-injections "${NUM_INJECTIONS}" \
    --sampler "${SAMPLER}" \
    --pe-run-type "${PE_RUN_TYPE}" \
    --nlive "${NLIVE}" \
    --threshold "${THRESHOLD}" \
    --nessai-npool "${NESSAI_NPOOL}" \
    --pytorch-threads "${PYTORCH_THREADS}" \
    --quiet

EXIT_CODE=$?

echo "============================================================"
echo "Finished inference job"
echo "Exit code: ${EXIT_CODE}"
echo "Date: $(date)"
echo "============================================================"

exit "${EXIT_CODE}"