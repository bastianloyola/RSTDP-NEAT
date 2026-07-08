#!/bin/bash

BASE_DIR=~/EvaluacionRedes/NEAT/estacionario/cartpole

for dir in "$BASE_DIR"/*/; do
    carpeta=$(basename "$dir")
    screen_name="NEATestcart${carpeta}"

    echo "Creando screen $screen_name"

    screen -S "$screen_name" -dm bash -c "
        cd '$dir' &&
        source ~/.bashrc &&
        eval \"\$(conda shell.bash hook)\" &&
        conda activate annarchy &&
        export LD_LIBRARY_PATH=\$HOME/local/lib:\$LD_LIBRARY_PATH &&
        python3 force-rstdp.py;
        exec bash
    "
done

echo 'Screens creados:'
screen -ls
