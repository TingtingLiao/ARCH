#!/bin/bash
NUM_VIEWS=$1
MODE=$2
PART=$3

PYTHON_SCRIPT="./render_animation.py"

if [[ $MODE == "gen" ]]; then
    echo "processing all the subjects"
    # render all the subjects
    NUM_THREADS=12
    cat ../data/mvphuman/$PART.txt | shuf | xargs -P$NUM_THREADS -I {} python $PYTHON_SCRIPT -s {} -r $NUM_VIEWS
fi

if [[ $MODE == "debug" ]]; then
    echo "Debug renderer"
    # render only one subject

    SUBJECT="222952"
    echo "Rendering $SUBJECT"
    python $PYTHON_SCRIPT -s $SUBJECT -r $NUM_VIEWS
fi
# bash render_single 360 gen train