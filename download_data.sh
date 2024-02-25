#!/bin/bash

DATA_DIR="data"

if [ ! -d "$DATA_DIR" ]; then
    mkdir $DATA_DIR
fi

cd $DATA_DIR

wget -r -N -c -np https://physionet.org/files/challenge-2017/1.0.0/

echo "Download complete."