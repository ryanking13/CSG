import pathlib

from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T
import pytorch_lightning ad pl

from .augmentations import TwoCropsTransform, RandAugment, augment_list

class Transforms:
    default = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    rand_augment = T.Compose([
        RandAugment(1, 6., augment_list),
        default,
    ])

    @classmethod
    def get_transform(cls, transform_type):
        try:
            return getattr(cls, transform_type)
        except AttributeError:
            print(f"Invalid transform: {transform_type}, using default transform.")
            return cls.default


class VISDA17DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        transforms="default",
        drop_last=True,
        siamese=True,
    )
        """
        Args:
            data_dir: visda17 dataset directory
            batch_size: batch size
            transforms: augmentations to use, if not given, use default augmentation setting
            drop_last: drop last incomplete training batch
            siamese: make train dataloader yield two images for training siamese network
        """
        super().__init__()
        self.data_dir = pathlib.Path(data_dir)
        self.batch_size = batch_size
        self.siamese = siamese
        self.drop_last = drop_last

        if isinstance(transforms, str):
            self.train_trainsforms = Transforms.get_transform(transforms)
        else:
            self.train_transforms = transforms
        
        print(f"Using transforms: {transforms}")

        self.val_transforms = Transforms.default

        if self.siamese:
            self.train_transforms = TwoCropsTransform(self.train_transforms)
        
    def setup(self, stage):
        self.trainset = datasets.ImageFolder(
            (self.data_dir / "train").as_posix(),
            self.train_transforms,
        )

        self.valset = datasets.ImageFolder(
            (self.data_dir / "validation").as_posix(),
            self.val_transforms,
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=20,
            pin_memory=True,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=20,
            pin_memory=True,
        )