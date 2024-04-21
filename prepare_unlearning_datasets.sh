#!/bin/bash

NAME="agpl3_python_2023-03-27-21-21-29"
EXT=".py"
python merge_source_files_into_json.py Prompts/prompts_${NAME}/ Source/source_${NAME}/ $EXT data/unlearning/train/${NAME}.json