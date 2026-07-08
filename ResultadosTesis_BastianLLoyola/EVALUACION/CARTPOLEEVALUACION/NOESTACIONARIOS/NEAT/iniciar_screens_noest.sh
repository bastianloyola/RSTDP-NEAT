#!/bin/bash

BASE_DIR=/home/DIINF/bloyola/EvaluacionRedes/NEAT/noestacionario/cartpole
PYTHON=/home/DIINF/bloyola/miniconda3/envs/annarchy/bin/python

for dir in "$BASE_DIR"/*/; do
    carpeta=$(basename "$dir")
    screen_name="NEATnoestcart${carpeta}"

    echo "Creando screen $screen_name"

    screen -S "$screen_name" -dm bash -c "
        cd '$dir' &&
        export LD_LIBRARY_PATH=\$HOME/local/lib:\$LD_LIBRARY_PATH &&
        $PYTHON ../force-rstdp.py > output.log 2>&1;
        exec bash
    "
done

echo "Screens creados:"
screen -ls
