#!/bin/bash

#SBATCH -J openfold_train       # job name
#SBATCH -C gpu
#SBATCH -q preempt
#SBATCH --ntasks-per-node=4
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --signal=B:USR1@60  # sig_time should match your checkpoint overhead time
#SBATCH --requeue
#SBATCH --open-mode=append

export NUM_NODES=${1}
export NUM_GPUS=${2}
export CHECKPOINT_PATH=$CFS/m3562/users/vidyagan/openfold/openfold/resources/openfold_params/finetuning_5.pt
export EXPERIMENT_NAME=test
export RESUME_MODEL_WEIGHTS_ONLY=False
export WANDB_ID=None
echo "jobstart $(date)";pwd

srun --nodes $NUM_NODES --gpus $NUM_GPUS --ntasks-per-node=4 python3 train_openfold.py $TRAIN_DATA_DIR $DATA_DIR/alignment_data/alignments $TEMPLATE_MMCIF_DIR $OUTPUT_DIR 2021-10-10 --train_chain_data_cache_path $DATA_DIR/pdb_data/data_caches/chain_data_cache.json --template_release_dates_cache_path $DATA_DIR/pdb_data/data_caches/mmcif_cache.json --config_preset finetuning --seed 42 --obsolete_pdbs_file_path $DATA_DIR/pdb_data/obsolete.dat --num_nodes $NUM_NODES --gpus 4 --precision bf16-mixed --resume_from_ckpt $CHECKPOINT_PATH --resume_model_weights_only $RESUME_MODEL_WEIGHTS_ONLY --use_experimental_loss True --log_performance False --wandb --experiment_name ${EXPERIMENT_NAME} --wandb_id $WANDB_ID --wandb_project ExperimentalFold --wandb_entity vganapati-lawrence-berkeley-national-laboratory --log_every_n_steps 1 --log_lr --checkpoint_every_epoch --max_epochs 100

echo "jobend $(date)";pwd