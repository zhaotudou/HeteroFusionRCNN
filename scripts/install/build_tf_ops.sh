#!/bin/bash
ops="bev_iou cropping grouping interpolate sampling"
for op in $ops;
do
    echo "compile ${op}"
    cd ${op} && bash tf_${op}_compile.sh && cd ..
done
