# Data Retrieval

- Data extractors to process RGB videos
- Extractors outputs `.ske` files for human skeleton coordinates and labels
- Extractors also outputs files (format to be defined) for objects coordinates and labels

## Dependencies
- Conda (25.7.0)


**Install Conda:**
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

bash Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/miniconda3

$HOME/miniconda3/bin/conda init zsh
```

## Creating and Activating Conda Environment

```bash
conda create --name openmmlab python=3.9.16 -y
conda activate openmmlab
```

```bash
# accept terms and services
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### Installing OpenMM

#### Linux
```bash
# Create clean conda environment
conda create -n masters python=3.10 -y
conda activate masters

# Install PyTorch (CPU only)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

# Install OpenMMLab package manager
pip install -U openmim

# Install matching versions of mmcv and mmdet
mim install "mmcv==2.1.0"
mim install "mmdet==3.2.0"

# Test instalation
python -c "import torch, mmcv, mmdet; print(torch.__version__, mmcv.__version__, mmdet.__version__)"
```

#### Getting Faster-RCNN
```bash
mkdir -p configs/faster_rcnn
wget https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py -P configs/faster_rcnn/

# Also grab the _base_ configs it depends on
mkdir -p configs/_base_/models configs/_base_/datasets configs/_base_/schedules configs/_base_
wget https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/_base_/models/faster-rcnn_r50_fpn.py -P configs/_base_/models/
wget https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/_base_/datasets/coco_detection.py -P configs/_base_/datasets/
wget https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/_base_/schedules/schedule_1x.py -P configs/_base_/schedules/
wget https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/_base_/default_runtime.py -P configs/_base_/

```