#!/bin/bash

# Exit immediately if any command fails
set -e

# Activate your specific environment
conda activate phd

# Run your script with arguments
python3 optimize.py --model-config modelconfig-running.json --dummy
