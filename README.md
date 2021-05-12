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
- 1x RTX 3070
  
To install Pytorch Lightning and Lightning bolts,

```sh
pip install pytorch-lightning lightning-bolts
```

## Classification (VisDA17)
### Dataset Setup

Download [VisDA17 dataset from here](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) or,
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

Simply run,

```
python run.py
```

or with options,

```sh
usage: run.py [-h] [-o OUTPUT] [-r ROOT] [-e EPOCHS] [-lr LEARNING_RATE] [-bs BATCH_SIZE] [-wd WEIGHT_DECAY]
              [--momentum MOMENTUM] [--num-classes NUM_CLASSES] [--emb-dim EMB_DIM] [--single-network]
              [--nce-weight NCE_WEIGHT] [--eval-only] [--num-gpus NUM_GPUS] [--resume RESUME] [--dev-run]
              [--exp-name EXP_NAME] [--augmentation AUGMENTATION] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output directory where checkpoint and log will be saved (default: logs)
  -r ROOT, --root ROOT  Root directory of the VisDA17 dataset (default: datasets\visda17)
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs (default: 30)
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning rate (default: 0.0001)
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size (default: 32)
  -wd WEIGHT_DECAY, --weight-decay WEIGHT_DECAY
                        Weight decay (default: 0.0005)
  --momentum MOMENTUM   Optimizer momentum (default: 0.9)
  --num-classes NUM_CLASSES
                        Number of classes (default: 12)
  --emb-dim EMB_DIM     Feature dimension (default: 128)
  --single-network      Train with single network, for comparison
  --nce-weight NCE_WEIGHT
                        Weight of nce loss (default: 0.1)
  --eval-only           Do not train, evaluate model
  --num-gpus NUM_GPUS   Number of gpus to use (default: All GPUs in the machine)
  --resume RESUME       Resume from the given checkpoint
  --dev-run             Run small steps to test whether model is valid
  --exp-name EXP_NAME   Experiment name used for a log directory name
  --augmentation AUGMENTATION
                        Augmentations to use (default: rand_augment)
  --seed SEED           Random seed (default: 0)
```

__Evaluation__

```
python run.py --eval-only --resume https://github.com/ryanking13/CSG/releases/download/v0.1/csg_resnet101.ckpt
```

### Results

| Model                                   | Accuracy |
| --------------------------------------- | -------- |
| CSG (from paper)                        | 64.1     |
| CSG-lightning                           | 66.1     |

__Differences from official implementation__

- No LR scheduler
- No layerwise LR modification
- RandAugment augmentation types


## Semantic Segmentation

Not supported yet.