#/bin/bash
PYTHON=python3
CUDA_PATH=/usr/local/cuda
TF_LIB=$($PYTHON -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
PYTHON_VERSION=$($PYTHON -c 'import sys; print("%d.%d"%(sys.version_info[0], sys.version_info[1]))')
# TF_PATH=/data/ljh/anaconda2/envs/HeteroFusion/lib/python$PYTHON_VERSION/site-packages/tensorflow/include
# TF_PATH=/home/liangcheng/anaconda2/envs/pointrcnn/lib/python$PYTHON_VERSION/site-packages/tensorflow/include
TF_PATH=$TF_LIB/include
$CUDA_PATH/bin/nvcc bev_iou_g.cu -o bev_iou_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 bev_iou.cpp bev_iou_g.cu.o -o bev_iou_so.so -shared -fPIC -L$TF_LIB -ltensorflow_framework -I $TF_PATH/external/nsync/public/ -I $TF_PATH -I $CUDA_PATH/include -lcudart -L $CUDA_PATH/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0
