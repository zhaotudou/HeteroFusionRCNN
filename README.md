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
git clone -b PointRCNN-seg ssh://git@git.sankuai.com/~lijiahong/heterofusion.git
```

####  Install Python dependencies

```
cd heterofusion
pip3 install -r requirements.txt
pip3 install tensorflow-gpu==1.6.0
```

#### Add `heterofusion (top level)` and `wavedata` to your PYTHONPATH

```
# For virtualenvwrapper users
add2virtualenv .
add2virtualenv wavedata
```

```
# For nonvirtualenv users
export PYTHONPATH=$PYTHONPATH:'/path/to/heterofusion'
export PYTHONPATH=$PYTHONPATH:'/path/to/heterofusion/wavedata'
```

#### Compile proto

Avod uses Protobufs to configure model and training parameters. Before the framework can be used, the protos must be compiled (from top level avod folder):

```
sh avod/protos/run_protoc.sh
```

Alternatively, you can run the `protoc` command directly:

```
protoc avod/protos/*.proto --python_out=.
```
#### Compile other tensorflow ops

**Note:** If compilation does not succeed, check the `TF_PATH` in `.sh` files to adpat to your environment setup. 

```
cd sampling
bash tf_sampling_compile.sh
```
```
cd cropping
bash tf_cropping_compile.sh
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

Alternatively, we provide a small dataset with few samples to validate your developing code. The path to this small dataset is `avod/tests/datasets/Kitti/object/`.

## Training

First of all, we will use the following three config file:

  1. `avod/configs/label_seg_preprocessing/rpn_cars.config`
  2. `avod/configs/rpb_cars_alt_1.config`
  3. `avod/configs/avod_cars_alt_2.config`
  
Before running any scripts, check the `dataset_dir` field in those config files and make sure it sets to:

```
dataset_dir: '/path/to/dataset/Kitti/object'
```
### 1. Preprocessing

The training data needs to be pre-processed to generate point level label for the RPN. 

Note: This script does parallel processing with `num_[class]_children` processes for faster processing. This can also be disabled inside the script by setting `in_parallel` to `False`.

```bash
cd avod
python scripts/preprocessing/gen_label_segs.py 
```

Once this script is done, you should now have the following folders inside `avod/data`:

```
data
    label_clusters
    label_segs
```

### 2. Train part-1 network - RPN network
To start training, run the following:

```bash
python avod/experiments/run_training.py --pipeline_config=avod/configs/rpn_cars_alt_1.config  --device='0' --data_split='train'
```
### 3. Generate data for part-2 network - RCNN network

```bash
python avod/experiments/run_evaluation.py --pipeline_config=avod/configs/rpn_cars_alt_1.config --data_split='train'
```
After running, you should see `proposals_and_scores` and `proposals_info` folders under `avod/data/outputs/rpn_cars_alt_1/predictions/
`.
### 4. Train part-2 network - RCNN network
To start training, run the following:

```bash
python avod/experiments/run_training.py --pipeline_config=avod/configs/avod_cars_alt_2.config  --device='0' --data_split='train'
```

### (Optional) Viewing Results
All results should be saved in `avod/data/outputs`. Here you should see `proposals_and_scores` and `final_predictions_and_scores` results. To visualize these results, you can run `demos/show_predictions_2d.py`. The script needs to be configured to your specific experiments. The `scripts/offline_eval/plot_ap.py` will plot the AP vs. step, and print the 5 highest performing checkpoints for each evaluation metric at the moderate difficulty.
