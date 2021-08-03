from PIL import Image
import numpy as np
import torch
from torchvision.datasets import Cityscapes as _Cityscapes


class Cityscapes(_Cityscapes):
    def __init__(
        self,
        root,
        split="train",
        mode="fine",
        target_type="instance",
        transform=None,
        target_transform=None,
        transforms=None,
        joint_transforms=None,
    ):
        """
        Cityscapes with joint transform support.

        Place dataset as below.

        📂 cityscapes <`root`>
        ┣ 📂 leftimg8bit
        ┃ ┣ 📂 train
        ┃ ┃ 📂 val
        ┃ ┗ 📂 test
        ┣ 📂 labels
        ┃ ┣ 📂 train
        ┃ ┃ 📂 val
        ┗ ┗ 📂 test
        """
        super().__init__(root, split, mode, target_type, transform, target_transform, transforms)
        self.joint_transforms = joint_transforms
        self.ignore_label = 255
        self.id2trainid = {}
        for cityscapes_class in _Cityscapes.classes:
            self.id2trainid[cityscapes_class.id] = cityscapes_class.train_id
    
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        targets =[]
        for i, t in enumerate(self.target_type):
            if t != "semantic":
                raise Exception(f"Only semantic segmentation is supported (given: {t})")

            target = Image.open(self.targets[index][i])
            targets.append(target)
        
        mask = tuple(targets) if len(targets) > 1 else targets[0]
        mask = np.array(mask.convert("L"))

        mask_copy = mask.copy()
        for k, v in self.id2trainid.items():
            mask_copy[mask==k] = v
        
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        # Apply joint transforms to image and target
        if self.joint_transforms is not None:
            image, mask = self.joint_transforms(image, mask)
        
        mask = torch.tensor(np.array(mask))

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask
