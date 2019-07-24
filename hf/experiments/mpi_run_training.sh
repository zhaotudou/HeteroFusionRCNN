#!/bin/bash
if [ $# -lt 2 ];then
    echo "Usage:"
    echo "  $0 <pipeline_config> <cuda_visible_devices>"
    exit
fi

export CUDA_VISIBLE_DEVICES=$2
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OLD_IFS="$IFS"
IFS=","
arr=($CUDA_VISIBLE_DEVICES)
IFS="$OLD_IFS"
gpu_num=${#arr[@]}

mpirun --allow-run-as-root -np ${gpu_num} -H localhost:${gpu_num} \
        -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
        -mca pml ob1 -mca btl ^openib \
        python $DIR/run_training.py --pipeline_config=$1 --data_split='train'
