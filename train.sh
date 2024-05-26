#!/bin/bash

source D:/Anaconda/anaconda3/etc/profile.d/conda.sh
conda activate dgl

CONFIG_DIR='config'

for config_file in "$CONFIG_DIR"/*.yaml; do
    echo "Running experiment with config file:  $config_file"
    python train.py --config "$config_file"
done

echo "All experiments Done"