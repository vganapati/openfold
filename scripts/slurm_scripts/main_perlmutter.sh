#!/bin/bash

echo "Script start $(date)";pwd

source ~/env_openfold

export NUM_NODES=${1}
export NUM_SUBMISSIONS=${2}
export TOTAL_GPUS=$((${NUM_NODES}*4))
export CHECKPOINT_PATH=$CFS/m3562/users/vidyagan/openfold/openfold/resources/openfold_params/finetuning_5.pt # change for future submissions

export JOB_ID_ARRAY=() 


for ((i = 1; i <= NUM_SUBMISSIONS; i++)); do
    echo "Submitting job to train with $DATASET_ID"

    # This will be run if all the previous jobs do not complete successfully XXX Check to confirm that termination is a notok condition
    export PREVIOUS_JOBS=$(IFS=:; echo "${JOB_ID_ARRAY[*]}")
    export COMMAND_NOTOK="sbatch --dependency=afternotok:$PREVIOUS_JOBS -A $NERSC_GPU_ALLOCATION --nodes=$NUM_NODES --gpus=$TOTAL_GPUS --time=$TIME child_perlmutter.sh $NUM_NODES $TOTAL_GPUS"

    echo "COMMAND: $COMMAND_NOTOK"
    
    JOB_ID=$(eval "$COMMAND_NOTOK" | awk '{print $4}')
    JOB_ID_ARRAY+=("$JOB_ID")
    echo "Job ID submitted: $JOB_ID"
done

# This will be run (final train and test) after the previous jobs complete successfully/unsuccessfully
export COMMAND_ANY="sbatch --dependency=afterany:$JOB_ID -A $NERSC_GPU_ALLOCATION -N $NUM_NODES -n $NUM_NODES --time=$TIME $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID 0 $SAVE_INTERVAL $PNM $RING $NUM_NODES $USE_H5 $JOB_ID_ORIG $DATA_TYPE $USE_MASKS"
JOB_ID_ANY=$(eval "$COMMAND_ANY" | awk '{print $4}')
echo "Job ID analysis: $JOB_ID_ANY"
JOB_FINAL_ARRAY+=("$JOB_ID_ANY")

echo "JOB_ID_ARRAY"
echo "${JOB_ID_ARRAY[@]}"
echo

echo "Script end $(date)";pwd