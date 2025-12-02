#!/usr/bin/env bash
# Regenerate deep-learning training data for all ISAs:
#   - AVX, AVX2, AVX512 on x86 (local)
#   - SVE using the same configuration-space generators (conceptually ARM,
#     but still benchmarked on this x86 machine via LLVM/MLIR)
#
# This script does NOT talk to the ARM machine; it uses the existing
# generate_dl_training_data* scripts, extended to include SVE.
#
# Usage:
#   ./generate_data.sh
#   LLVM_PROJECT_PATH=/path/to/llvm-project ./generate_data.sh
#
# Notes:
#   - This can take a long time (tens of minutes to hours) depending on
#     matrix sizes, num_runs, and max_configs.
#   - You can edit NUM_RUNS_* and MATRIX_SIZES_* below to control workload.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

###############################################################################
# Configuration knobs
###############################################################################

# LLVM path (used implicitly by the Python scripts via LLVM_PROJECT_PATH)
export LLVM_PROJECT_PATH="${LLVM_PROJECT_PATH:-/home/mazaya/Documents/cmu/interviews/vorticity/llvm-project-vorticity}"

# How many runs per configuration
NUM_RUNS_BASE=5
NUM_RUNS_CORRECTED=5
NUM_RUNS_FORMALIZED=5
NUM_RUNS_REFINED=5

# Matrix sizes for the base script (M,K,N)
# Keep this small and simple; the richer remainder coverage comes from the
# corrected/formalized/refined generators, which already include many
# rectangular shapes and all remainder patterns.
MATRIX_SIZES_BASE=(
  "32,32,32"
)

# ISAs to include in the datasets.
# We explicitly include SVE, but NOT SME (not supported end-to-end yet).
# VECTOR_ISAS_ALL=("AVX" "AVX2" "AVX512" "SVE")
VECTOR_ISAS_ALL=("AVX512")

###############################################################################
# Helper: join array by space
###############################################################################

join_by_space() {
  local IFS=" "
  echo "$*"
}

VECTOR_ISAS_ARG=$(join_by_space "${VECTOR_ISAS_ALL[@]}")
MATRIX_SIZES_ARG=$(join_by_space "${MATRIX_SIZES_BASE[@]}")

echo "============================================================"
echo "Deep Learning Data Generation - All ISAs (AVX/AVX2/AVX512/SVE)"
echo "============================================================"
echo "Project root        : $ROOT_DIR"
echo "LLVM_PROJECT_PATH   : $LLVM_PROJECT_PATH"
echo "Vector ISAs         : ${VECTOR_ISAS_ALL[*]}"
echo

###############################################################################
# 1) Base input space (original generate_dl_training_data.py)
###############################################################################

echo "[1/4] Base input space (generate_dl_training_data.py)"
echo "------------------------------------------------------------"
echo "Matrix sizes : ${MATRIX_SIZES_BASE[*]}"
echo "Vector ISAs  : ${VECTOR_ISAS_ALL[*]}"
echo "Num runs     : $NUM_RUNS_BASE"
echo

python3 generate_dl_training_data.py \
  --output-dir dl_training_data_all_isas \
  --num-runs "$NUM_RUNS_BASE" \
  --matrix-sizes $MATRIX_SIZES_ARG \
  --vector-isas $VECTOR_ISAS_ARG

echo

###############################################################################
# 2) Corrected input space
###############################################################################

echo "[2/4] Corrected input space (generate_dl_training_data_corrected.py)"
echo "--------------------------------------------------------------------"
echo "Vector ISAs  : ${VECTOR_ISAS_ALL[*]}"
echo "Num runs     : $NUM_RUNS_CORRECTED"
echo

python3 generate_dl_training_data_corrected.py \
  --output-dir dl_training_data_corrected_all_isas \
  --num-runs "$NUM_RUNS_CORRECTED" \
  --vector-isas $VECTOR_ISAS_ARG

echo

###############################################################################
# 3) Formalized input space
###############################################################################

echo "[3/4] Formalized input space (generate_dl_training_data_formalized.py)"
echo "---------------------------------------------------------------------"
echo "Vector ISAs  : ${VECTOR_ISAS_ALL[*]}"
echo "Num runs     : $NUM_RUNS_FORMALIZED"
echo

python3 generate_dl_training_data_formalized.py \
  --output-dir dl_training_data_formalized_all_isas \
  --num-runs "$NUM_RUNS_FORMALIZED" \
  --vector-isas $VECTOR_ISAS_ARG

echo

###############################################################################
# 4) Refined input space
###############################################################################

echo "[4/4] Refined input space (generate_dl_training_data_refined.py)"
echo "----------------------------------------------------------------"
echo "Vector ISAs  : ${VECTOR_ISAS_ALL[*]}"
echo "Num runs     : $NUM_RUNS_REFINED"
echo

python3 generate_dl_training_data_refined.py \
  --output-dir dl_training_data_refined_all_isas \
  --num-runs "$NUM_RUNS_REFINED" \
  --vector-isas $VECTOR_ISAS_ARG

echo
echo "============================================================"
echo "Data generation completed."
echo "Outputs:"
echo "  - dl_training_data_all_isas/"
echo "  - dl_training_data_corrected_all_isas/"
echo "  - dl_training_data_formalized_all_isas/"
echo "  - dl_training_data_refined_all_isas/"
echo "============================================================"
echo


