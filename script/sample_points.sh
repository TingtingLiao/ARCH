#!/bin/bash
MODE=$1
PART=$2

PYTHON_SCRIPT="./preprocess.py"

if [[ $MODE == "gen" ]]; then
    echo "processing all the subjects"
    NUM_THREADS=12
    cat ../data/mvphuman/$PART.txt | xargs -P$NUM_THREADS -I {} python $PYTHON_SCRIPT -s {}
fi

if [[ $MODE == "debug" ]]; then
    echo "Debug renderer"
    # render only one subject
    SUBJECT="222952"
    echo "Sampling point mvphuman $SUBJECT"
    python $PYTHON_SCRIPT -s $SUBJECT
fi
# bash sample_points.sh gen train