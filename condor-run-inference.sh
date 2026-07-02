#!/usr/bin/env bash
set -eo pipefail

LABEL="${1:-ml2ml-paper-1}"
INJECTION_INDEX="${2:-0}"
SAMPLER="${3:-pocomc}"
PE_RUN_TYPE="${4:-ml2ml}"
DISTANCE="${5:-400}"
NLIVE="${6:-1024}"
THRESHOLD="${7:-0.2}"
NPOOL="${8:-4}"
PYTORCH_THREADS="${9:-1}"

echo "============================================================"
echo "Starting inference job"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "PWD: $(pwd)"
echo "LABEL: ${LABEL}"
echo "INJECTION_INDEX: ${INJECTION_INDEX}"
echo "SAMPLER: ${SAMPLER}"
echo "PE_RUN_TYPE: ${PE_RUN_TYPE}"
echo "DISTANCE: ${DISTANCE}"
echo "NLIVE: ${NLIVE}"
echo "THRESHOLD: ${THRESHOLD}"
echo "NPOOL: ${NPOOL}"
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

export PATH="/home/suyog.garg/.conda/envs/phd/bin:/usr/local/bin:/usr/bin:/bin"

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

echo "Checking core imports..."
"${PYTHON_EXE}" -c "import sys; print(sys.executable)"
"${PYTHON_EXE}" -c "import torch; print('torch', torch.__version__)"
"${PYTHON_EXE}" -c "import bilby; print('bilby', bilby.__version__)"
"${PYTHON_EXE}" -c "import nessai; print('nessai import ok')"

echo "Running inference.py..."

"${PYTHON_EXE}" inference.py --run-one-injection --quiet \
    --label "${LABEL}" \
    --injection-index "${INJECTION_INDEX}" \
    --sampler "${SAMPLER}" \
    --pe-run-type "${PE_RUN_TYPE}" \
    --distance-factor "${DISTANCE}" \
    --nlive "${NLIVE}" \
    --threshold "${THRESHOLD}" \
    --npool "${NPOOL}" \
    --pytorch-threads "${PYTORCH_THREADS}" \
    --model-name flexcvae-model-backup-20260619-064140-epoch98.pt \
    --model-config modelconfig-flexcvae-20260619-064140.json \
    --calmodel-name calibrator_model_20260623-010953_epoch74.pt

EXIT_CODE=$?

echo "============================================================"
echo "Finished inference job for label ${LABEL} and injection index ${INJECTION_INDEX} with sampler ${SAMPLER} and PE run type ${PE_RUN_TYPE}."
echo "Exit code: ${EXIT_CODE}"
echo "Date: $(date)"
echo "============================================================"

exit "${EXIT_CODE}"

mkdir -p /home/suyog.garg/phd-main/umami/logs
mv /home/suyog.garg/phd-main/umami/inference_${CONDOR_CLUSTER_ID}_${CONDOR_PROC_ID}.* /home/suyog.garg/phd-main/umami/logs/ 2>/dev/null || true