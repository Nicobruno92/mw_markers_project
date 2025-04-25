#!/bin/bash
#SBATCH --job-name=permutation
#SBATCH --output=logs/permutation_%A_%a.out
#SBATCH --error=logs/permutation_%A_%a.err
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --chdir=/Users/nicolas.bruno/mw_markers_project
#SBATCH --array=0-3              # 4 comparisons
#SBATCH --mail-type=END,FAIL
#SBATCH --hint=nomultithread

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules
module load proxy 2>/dev/null || echo "Module proxy not available, continuing..."

# Set working directory to project root
cd /Users/nicolas.bruno/mw_markers_project

# Set PYTHONPATH to include all necessary directories
export PYTHONPATH="${PYTHONPATH}:/Users/nicolas.bruno/mw_markers_project"

# Try to activate the ML environment
echo "Attempting to activate Python environment..."

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    echo "Using miniconda3 path"
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate ML
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    echo "Using anaconda3 path"
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate ML
elif command -v conda >/dev/null 2>&1; then
    echo "Using conda from PATH"
    eval "$(conda shell.bash hook)"
    conda activate ML
else
    echo "Using module method"
    module load anaconda3 || module load miniconda3 || module load conda || echo "No conda module found"
    conda activate ML || echo "Failed to activate ML environment"
fi

# Define the comparisons to run
COMPARISONS=("on-task_vs_mw" "on-task_vs_dMW" "on-task_vs_sMW" "dMW_vs_sMW")

# Get the current comparison based on SLURM array task ID
COMPARISON=${COMPARISONS[$SLURM_ARRAY_TASK_ID]}

# Set up unique database file for each job to avoid SQLite conflicts
DB_SUFFIX=$(date +"%Y%m%d_%H%M%S")_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
DEST_DB="multivariate_merf_full_permutations_${DB_SUFFIX}.db"

echo "Starting permutation analysis for comparison: $COMPARISON"
echo "Using database: $DEST_DB"
echo "Running on $(hostname) with $SLURM_CPUS_PER_TASK CPUs"
echo "Start time: $(date)"

# Create a temporary Python script for this specific job
TEMP_SCRIPT="temp_run_permutation_${SLURM_ARRAY_TASK_ID}.py"

cat > $TEMP_SCRIPT << EOF
#!/usr/bin/env python
import os
import sys
sys.path.insert(0, '../')
sys.path.insert(0, './')

# Add the src directory to the path
sys.path.insert(0, 'src')

from src.MVPA.full_optim_permutations import run_full_permutation_analysis

# Run the analysis
run_full_permutation_analysis(
    comparison='${COMPARISON}',
    probe='PC', 
    k=4, 
    n_permutations=500,  
    n_trials_per_perm=50,
    source_db='multivariate_merf_study_paper.db',
    source_study_suffix='_paper',
    dest_db='${DEST_DB}'
)
EOF

# Run the temporary Python script
echo "Executing Python script..."
python $TEMP_SCRIPT

# Clean up
rm $TEMP_SCRIPT

echo "Permutation analysis for $COMPARISON completed"
echo "End time: $(date)"

# Optional: Combine results from different databases at the end if needed
# This would need to be done in a separate step after all jobs complete

exit 0 