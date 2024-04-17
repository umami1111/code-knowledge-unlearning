#!/bin/bash

SRC="source_agpl3_python_2023-03-27-21-21-29"
EXT=".py"
python merge_source_files_into_json.py Source/${SRC}/ $EXT data/unlearning/train/${SRC}.json