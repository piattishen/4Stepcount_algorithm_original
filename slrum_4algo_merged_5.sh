#!/bin/bash
# =============================================================================
# slrum_4algo_merged.sh - SLURM array job for the merged four-algorithm
# cluster pipeline (pipeline_cluster_v3.py).
#
# Per-algorithm sensor location filter (applied inside Python):
#   adept     -> any
#   oak       -> any
#   oxford    -> wrist,hip
#   verisense -> wrist
#
# Each array task = one subject's accel/ directory.
#
# Submit:
#   sbatch slrum_array/slrum_4algo_merged.sh
# Resubmit one failed subject:
#   sbatch --array=7 slrum_array/slrum_4algo_merged.sh
# =============================================================================
#SBATCH -J 4algo_v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --array=1-5
#SBATCH --output=logs/4algo_merged_%A_%a.out
#SBATCH --error=logs/4algo_merged_%A_%a.err

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_INPUT=/scratch/wang.yichen8/PAAWS_FreeLiving
BASE_OUTPUT=/scratch/wang.yichen8/PAAWS_results_v3
PIPELINE=/home/wang.yichen8/pipeline_algorithm_stepcount/pipeline_cluster_v3.py

subjects=(
    DS_49  DS_51  DS_58   DS_59   DS_87  DS_48
)

subj=${subjects[$SLURM_ARRAY_TASK_ID - 1]}
INPUT_DIR="${BASE_INPUT}/${subj}/accel"
OUTPUT_DIR="${BASE_OUTPUT}/${subj}"

echo "================================================"
echo "Task ${SLURM_ARRAY_TASK_ID}: subject=${subj}"
echo "Input dir   : ${INPUT_DIR}"
echo "Output dir  : ${OUTPUT_DIR}"
echo "================================================"

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: missing input dir $INPUT_DIR"; exit 1
fi

mkdir -p "$OUTPUT_DIR" logs
module load singularity
module load anaconda3/2024.06
source activate stepcount

# Optional override - only set if you move the ADEPT R sources
# export ADEPT_R_DIR=/home/wang.yichen8/4_algo_original_code/ADEPT

python "$PIPELINE" "$INPUT_DIR" "$OUTPUT_DIR" \
    --algorithms verisense adept oak oxford \
    --interval 5 \
    --n-cores ${SLURM_CPUS_PER_TASK:-16}

echo "Done: $subj"
