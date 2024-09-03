#!/bin/bash

echo "Script start $(date)";pwd

source ~/env_openfold

export NUM_NODES=${1}
export NUM_SUBMISSIONS=${2}
export NERSC_GPU_ALLOCATION=${3}
export WANDB_ENTITY=${4}
export TIME=${5}
export CHECKPOINT_PATH=${6} # initial path, changes for future submissions

export EXPERIMENT_NAME=EXPR_001
export WANDB_ID=ID_001
export WANDB_PROJECT=ExperimentalFold

export TOTAL_GPUS=$((${NUM_NODES}*4))
export RESUME_MODEL_WEIGHTS_ONLY=True # changes for future submissions
export JOB_ID_ARRAY=() 

for ((i = 1; i <= NUM_SUBMISSIONS; i++)); do
    echo "Submitting job to train:"

    # This will be run after all the previous jobs have completed
    export PREVIOUS_JOBS=$(IFS=:; echo "${JOB_ID_ARRAY[*]}")

    export COMMAND_ANY="sbatch "

    if [ -n "$PREVIOUS_JOBS" ]; then
        COMMAND_ANY+="--dependency=afterany:$PREVIOUS_JOBS "
    fi

    COMMAND_ANY+="-A $NERSC_GPU_ALLOCATION --nodes=$NUM_NODES --gpus=$TOTAL_GPUS --time=$TIME scripts/slurm_scripts/child_perlmutter.sh $NUM_NODES $TOTAL_GPUS $CHECKPOINT_PATH $RESUME_MODEL_WEIGHTS_ONLY $WANDB_ENTITY $EXPERIMENT_NAME $WANDB_ID $WANDB_PROJECT"

    echo "COMMAND: $COMMAND_ANY"
    
    JOB_ID=$(eval "$COMMAND_ANY" | awk '{print $4}')
    JOB_ID_ARRAY+=("$JOB_ID")
    echo "Job ID submitted: $JOB_ID"
    export CHECKPOINT_PATH=LATEST
    export RESUME_MODEL_WEIGHTS_ONLY=False
done

echo "Script end $(date)";pwd