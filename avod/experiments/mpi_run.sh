#!/bin/bash


export PYTHONPATH=":/chenzhuo/heterofusion:/chenzhuo/heterofusion/wavedata"
export CUDA_VISIBLE_DEVICES="4,5,6,7"
gpu_num=4

mpirun --allow-run-as-root -np ${gpu_num} -H localhost:${gpu_num} \
        -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
        -mca pml ob1 -mca btl ^openib \
        python run_training.py --pipeline_config=../configs/rpn_cars_alt_1.config