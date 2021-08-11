import torchvision.transforms as T

from . import joint_transforms
from .rand_augment import RandAugment


class Transforms:
    @classmethod
    def get_transform(cls, transform_type):
        try:
            return getattr(cls, transform_type)
        except AttributeError:
            print(f"Invalid transform: {transform_type}, using default transform.")
            return cls.default


class ClassificationTransforms(Transforms):
    default = T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    rand_augment = T.Compose(
        [
            RandAugment(1, 6.0),
            default,
        ]
    )


class SegmentationTransforms(Transforms):

    default = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    joint_default = joint_transforms.Compose(
        [
            joint_transforms.RandomSizeAndCrop(
                512,
                crop_nopad=False,
                scale_min=0.75,
                scale_max=1.25,
            ),
            joint_transforms.Resize(512),
            joint_transforms.RandomHorizontallyFlip(),
        ]
    )

    color_jitter = T.Compose(
        [
            T.ColorJitter(0.4, 0.4, 0.4, 0.4),
            default,
        ]
    )


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]
