#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
gpu_num=4

mpirun --allow-run-as-root -np ${gpu_num} -H localhost:${gpu_num} \
        -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
        -mca pml ob1 -mca btl ^openib \
        python $DIR/run_training.py --pipeline_config=$1 --data_split='train'
