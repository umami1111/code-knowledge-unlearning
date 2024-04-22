#!/bin/bash

# prepare dataset
NAME="agpl3x2000_python_2023-03-27-21-21-29"
EXT=".py"
python merge_source_files_into_json.py Prompts/prompts_${NAME}/ Source/source_${NAME}/ $EXT data/unlearning/train/${NAME}.json

# Train models
python run_unlearning.py --output_dir output/unlearning --per_device_train_batch_size 2 --learning_rate 5e-05 --weight_decay 0.1 --warmup_steps 0 --num_train_epochs 1 --data_dir data/unlearning
python run_unlearning.py --output_dir output/unlearning --per_device_train_batch_size 2 --learning_rate 5e-06 --weight_decay 0.1 --warmup_steps 0 --num_train_epochs 1 --data_dir data/unlearning

# Generate programs
python Example_Parrot.py output/unlearning/models/lr-5e-05_bs-2_accsteps-1_epochs-1.0_maxsteps-0_warmsteps-0_lora-8_seed-42/ prompts_agpl3x2000_python_2023-03-27-21-21-29 --output_prefix unlearn_lr-5e-5_epoch-1
python Example_Parrot.py output/unlearning/models/lr-5e-06_bs-2_accsteps-1_epochs-1.0_maxsteps-0_warmsteps-0_lora-8_seed-42/ prompts_agpl3x2000_python_2023-03-27-21-21-29 --output_prefix unlearn_lr-5e-6_epoch-1

# Evaluation
