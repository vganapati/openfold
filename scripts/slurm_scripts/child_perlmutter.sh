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


echo "jobstart $(date)";pwd

srun --nodes ${1} --gpus ${2} --ntasks-per-node=4 python3 train_openfold.py $TRAIN_DATA_DIR $DATA_DIR/alignment_data/alignments $TEMPLATE_MMCIF_DIR $OUTPUT_DIR 2021-10-10 --train_chain_data_cache_path $DATA_DIR/pdb_data/data_caches/chain_data_cache.json --template_release_dates_cache_path $DATA_DIR/pdb_data/data_caches/mmcif_cache.json --config_preset finetuning --seed 42 --obsolete_pdbs_file_path $DATA_DIR/pdb_data/obsolete.dat --num_nodes $NUM_NODES --gpus 4 --precision bf16-mixed --resume_from_ckpt $CHECKPOINT_PATH --resume_model_weights_only True --use_experimental_loss True

echo "jobend $(date)";pwd