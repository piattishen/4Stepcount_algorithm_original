#!/bin/bash
#SBATCH -J Stepcount_Combined          # Job name
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                      # One task per array job
#SBATCH --cpus-per-task=32              # CPU cores for R parallel (ADEPT mclapply)
#SBATCH --time=08:00:00                # Walltime (increase if needed for large datasets)
#SBATCH --mem=64G                      # Memory (OAK/Verisense are memory-intensive)
#SBATCH --array=1-20                   # One task per subject (adjust range to match subject count)
#SBATCH --output=logs/stepcount_%A_%a.out   # stdout log per task
#SBATCH --error=logs/stepcount_%A_%a.err    # stderr log per task

#  Paths  edit these 
BASE_INPUT="/scratch/wang.yichen8/PAAWS_FreeLiving"
BASE_OUTPUT="/scratch/wang.yichen8/PAAWS_results"
PIPELINE="/home/wang.yichen8/pipeline_algorithm_stepcount/pipeline_cluster.py"

#  Subject list — one entry per array task
#  Count the subjects and set --array=1-N above to match
subjects=(
    DS_20  DS_138  DS_139  DS_140  DS_235  DS_239  DS_240  DS_246
    DS_36  DS_37   DS_38   DS_39   DS_42   DS_44   DS_48   DS_49
    DS_51  DS_58   DS_59   DS_87
    # Add more subjects here; update --array range to match
)

#  Get subject for this array task (1-indexed)
subj=${subjects[$SLURM_ARRAY_TASK_ID - 1]}

INPUT_DIR="${BASE_INPUT}/${subj}/accel"
# Output goes to BASE_OUTPUT/<subject>/ so each subject has its own folder.
# Example result: /scratch/wang.yichen8/PAAWS_results/DS_138/DS_138-Free-RightWaist_verisense_steps.csv
OUTPUT_DIR="${BASE_OUTPUT}/${subj}"

echo "================================================"
echo "SLURM Array Task : $SLURM_ARRAY_TASK_ID"
echo "Subject          : $subj"
echo "Input dir        : $INPUT_DIR"
echo "Output dir       : $OUTPUT_DIR"
echo "================================================"

#  Check input exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory not found: $INPUT_DIR"
    exit 1
fi

#  Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

#  Load modules
module load anaconda3/2024.06
module load singularity        # needed for ADEPT (R via Rocker container)
source activate stepcount

#  Run four-algorithm pipeline
#  INPUT_DIR is a directory — pipeline_cluster.py processes all *.csv inside it.
python "$PIPELINE" "$INPUT_DIR" "$OUTPUT_DIR" \
    --algorithms verisense adept oak oxford \
    --interval 5

echo "Done: $subj"