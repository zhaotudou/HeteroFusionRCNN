## Environment setup

```
Ubuntu: 16.06
Python: 3.6
Tensorflow: 1.6.0
Nvidia driver: 410.78
CUDA: 9.0
```

#### Clone this repo

```
git clone -b HeteroFusion ssh://git@git.sankuai.com/~lijiahong/heterofusion.git
```

####  Install Python dependencies

```
cd heterofusion
pip3 install -r requirements.txt
pip3 install tensorflow-gpu==1.6.0
```

#### Add `heterofusion (top level)` to your PYTHONPATH

```
# For virtualenvwrapper users
add2virtualenv .
```

```
# For nonvirtualenv users
export PYTHONPATH=$PYTHONPATH:'/path/to/heterofusion'
```

#### Compile proto

HeteroFusion uses Protobufs to configure model and training parameters. Before the framework can be used, the protos must be compiled (from top level heterofusion folder):

```
bash hf/protos/run_protoc.sh
```

Alternatively, you can run the `protoc` command directly:

```
protoc hf/protos/*.proto --python_out=.
```
#### Compile other tensorflow ops

**Note:** If compilation does not succeed, check the `TF_PATH` in `.sh` files to adpat to your environment setup. 

```
bash scripts/install/build_tf_ops.sh 
```
## Dataset
To train on the [Kitti Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d):

- Download the data and place it in your home folder at `/path/to/dataset/Kitti/object`
- Go [here](https://drive.google.com/open?id=1yjCwlSOfAZoPNNqoMtWfEjPCfhRfJB-Z) and download the `train.txt`, `val.txt` and `trainval.txt` splits into `~/Kitti/object`. 

The folder should look something like the following:

```
Kitti
    object
        testing
        training
            calib
            image_2
            label_2
            velodyne
        train.txt
        val.txt
```

Alternatively, we provide a small dataset with few samples to validate your developing code. The path to this small dataset is `hf/tests/datasets/Kitti/object/`.

## Training

First of all, we will use the following three config file:

  1. `hf/configs/rpn_multiclass.config`
  2. `hf/configs/rcnn_multiclass.config`
  
Before running any scripts, check the `dataset_dir` field in those config files and make sure it sets to:

```
dataset_dir: '/path/to/dataset/Kitti/object'
```
### 1. Train stage-1 network - RPN network
To start training, run the following (single-GPU version):

```bash
python hf/experiments/run_training.py --pipeline_config=hf/configs/rpn_multiclass.config  --device='0' --data_split='train'
```

or the multi-GPU version:
```bash
bash hf/experiments/mpi_run_training.sh hf/configs/rpn_multiclass.config  '0,1,2,3'
```
### 2. Generate data for stage-2 network - RCNN network

```bash
python hf/experiments/run_evaluation.py --pipeline_config=hf/configs/rpn_multiclass.config --data_split='train' --for_rcnn_train --save_rpn_feature
```
After running, you should see `proposals_and_scores`, `proposals_iou` and `rpn_feature` folders under `hf/data/outputs/rpn_multiclass/predictions_for_rcnn_train/
`.
### 3. Train stage-2 network - RCNN network
To start training, run the following (single-GPU version):

```bash
python hf/experiments/run_training.py --pipeline_config=hf/configs/rcnn_multiclass.config  --device='0' --data_split='train'
```
or the multi-GPU version:
```bash
bash hf/experiments/mpi_run_training.sh hf/configs/rcnn_multiclass.config  '0,1,2,3'
```

### (Optional) Viewing Results
All results should be saved in `hf/data/outputs`. Here you should see `proposals_and_scores` and `final_predictions_and_scores` results. To visualize these results, you can run `demos/show_predictions_2d.py` or `demos/show_predictions_3d.py`. The script needs to be configured to your specific experiments. The `scripts/offline_eval/plot_ap.py` will plot the AP vs. step, and print the 5 highest performing checkpoints for each evaluation metric at the moderate difficulty.
