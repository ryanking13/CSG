# CSG-lightning

Unofficial Pytorch Lightning implementation of Contrastive Syn-to-Real Generalization (ICLR 2021).

Based on:

- [Original Paper](https://arxiv.org/abs/2104.02290)
- [Official CSG Pytorch Implementation](https://github.com/NVlabs/CSG)
## Environment Setup

Tested in a Python 3.8 environment in Linux and Windows with:

- Pytorch: 1.8.1
- [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning): 1.3.1
- [Lightning bolts](https://github.com/PyTorchLightning/lightning-bolts): 0.3.3
- [Torchmetrics](https://github.com/PyTorchLightning/metrics): 0.3.2
- 1x RTX 3070
  
Installing the dependencies:

```sh
pip install pytorch-lightning lightning-bolts torchmetrics
```

## Classification (VisDA17)
### Dataset Setup

Download [VisDA17 dataset from official website](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) or,
use the provided script for your convenience.

```sh
# The script downloads and extracts VisDA17 dataset.
# Note: It takes a very long time to download full dataset.
python datasets/prepare_visda17.py
```

If you downloaded the dataset manually, extract and place them as below.

```
📂 datasets
 ┣ 📂 visda17
 ┃ ┣ 📂 train
 ┃ ┃ 📂 validation
 ┗ ┗ 📂 test
```

### How to run

__Training__

Simply run:

```sh
python run.py
```

or with options,

```sh
usage: run.py [-h] [-o OUTPUT] [-r ROOT] [-e EPOCHS] [-lr LEARNING_RATE] [-bs BATCH_SIZE] [-wd WEIGHT_DECAY] [--task {classification,segmentation}] [--encoder {resnet101,deeplab50,deeplab101}] [--momentum MOMENTUM] [--num-classes NUM_CLASSES] [--eval-only] [--gpus GPUS]
              [--resume RESUME] [--dev-run] [--exp-name EXP_NAME] [--augmentation AUGMENTATION] [--seed SEED] [--fc-dim FC_DIM] [--no-apool] [--single-network] [--stages STAGES [STAGES ...]] [--emb-dim EMB_DIM] [--emb-depth EMB_DEPTH] [--num-patches NUM_PATCHES]
              [--moco-weight MOCO_WEIGHT] [--moco-queue-size MOCO_QUEUE_SIZE] [--moco-momentum MOCO_MOMENTUM] [--moco-temperature MOCO_TEMPERATURE]
```

__Evaluation__

```
python run.py --eval-only --resume https://github.com/ryanking13/CSG/releases/download/v0.2/csg_resnet101.ckpt
```

### Results

| Model            | Accuracy |
| ---------------- | -------- |
| CSG (from paper) | 64.1     |
| CSG (reimpl)     | 66.1     |

## Semantic Segmentation

### Dataset Setup (GTA5 ==> Cityscapes)

Download [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) and [Cityscapes](https://www.cityscapes-dataset.com/downloads/) datasets.

Place them as below.

```
📂 datasets
 ┣ 📂 GTA5
 ┃ ┣ 📂 images 
 ┃ ┃ ┣ 📜 00001.png
 ┃ ┃ ┣ ...
 ┃ ┃ ┗ 📜 24966.png
 ┃ ┃ ┣ 📂 labels
 ┃ ┃ ┣ 📜 00001.png
 ┃ ┃ ┣ ...
 ┃ ┃ ┗ 📜 24966.png
 ┣ 📂 cityscapes
 ┃ ┣ 📂 leftImg8bit
 ┃ ┃ ┣ 📂 train
 ┃ ┃ ┃ 📂 val
 ┗ ┗ ┗ 📂 test
 ┃ ┣ 📂 gtFine 
 ┃ ┃ ┣ 📂 train
 ┃ ┃ ┃ 📂 val
 ┗ ┗ ┗ 📂 test
```

### How to run

__Training__

Simply run:

```sh
./run_seg.sh
```

__Evaluation__

```
./run_seg --eval-only --resume https://github.com/ryanking13/CSG/releases/download/v0.2/csg_deeplab50.ckpt
```

### Results

| Model            | IoU   |
| ---------------- | ----- |
| CSG (from paper) | 35.27 |
| CSG (reimpl)     | 34.71 |

## Differences from official implementation

- Warmup LR scheduler
- No layerwise LR modification
- RandAugment augmentation types

## Known Issues

- I got error `Distributed package doesn't have NCCL built in` 

On windows, `nccl` is not supported, try:

```bat
set PL_TORCH_DISTRIBUTED_BACKEND=gloo
```

