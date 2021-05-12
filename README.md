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

```
python run.py
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