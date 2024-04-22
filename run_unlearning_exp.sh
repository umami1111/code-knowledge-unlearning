#!/bin/bash

# prepare dataset
NAME="agpl3x2000_python_2023-03-27-21-21-29"
EXT=".py"
python merge_source_files_into_json.py Prompts/prompts_${NAME}/ Source/source_${NAME}/ $EXT data/unlearning/train/${NAME}.json

# Train models
python run_unlearning.py --output_dir output/unlearning --per_device_train_batch_size 2 --learning_rate 5e-04 --weight_decay 0.1 --warmup_steps 0 --num_train_epochs 1 --data_dir data/unlearning
python run_unlearning.py --output_dir output/unlearning --per_device_train_batch_size 2 --learning_rate 5e-04 --weight_decay 0.1 --warmup_steps 0 --num_train_epochs 3 --data_dir data/unlearning

# Generate programs


# Evaluation
