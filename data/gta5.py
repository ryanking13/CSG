import pathlib

from PIL import Image
import numpy as np
import torch
from torchvision.datasets import Cityscapes as _Cityscapes


class GTA5:
    def __init__(
        self, root, transform=None, target_transform=None, joint_transforms=None
    ):
        """
        GTA5 (Playing for data) dataset.

        Place dataset as below.

        📂 GTA5 <`root`>
        ┣ 📂 images
        ┃ ┣ 📜 00001.png
        ┃ ┃ ...
        ┃ ┗ 📜 24966.png
        ┣ 📂 labels
        ┃ ┣ 📜 00001.png
        ┃ ┃ ...
        ┗ ┗ 📜 24966.png
        """

        self.root = pathlib.Path(root)
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "labels"

        self.imgs_path = sorted(self.image_dir.glob("*.png"))

        self.transform = transform
        self.target_transform = target_transform
        self.joint_transforms = joint_transforms

        self.images = [img.as_posix() for img in self.imgs_path]
        self.masks = [(self.mask_dir / img.name).as_posix() for img in self.imgs_path]

        self.ignore_label = 255
        self.color2trainid = {}
        for cityscapes_class in _Cityscapes.classes:
            if cityscapes_class.train_id in (-1, 255):
                continue
            self.color2trainid[cityscapes_class.color] = cityscapes_class.train_id

    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.masks[index]

        image = Image.open(img_path).convert("RGB")
        mask = np.asarray(Image.open(mask_path).convert("RGB"))

        # Convert color mask to trainid mask compatible to cityscapes
        mask_size = mask[:, :, 0].shape
        mask_copy = np.full(mask_size, self.ignore_label, dtype=np.uint8)

        for k, v in self.color2trainid.items():
            if v in (-1, 255):
                continue

            color_mask = mask == np.array(k)
            mask_copy[
                color_mask[:, :, 0] & color_mask[:, :, 1] & color_mask[:, :, 2]
            ] = v

        mask = Image.fromarray(mask_copy.astype(np.uint8))

        # GTA dataset contains some image/masks where their sizes' mismatch
        # Force resizing
        if image.size != mask.size:
            image = image.resize(mask.size, Image.BILINEAR)

        # Apply joint trainsforms to image/target
        if self.joint_transforms is not None:
            image, mask = self.joint_transforms(image, mask)

        mask = torch.tensor(np.array(mask))

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return image, mask

    def __len__(self):
        return len(self.images)
