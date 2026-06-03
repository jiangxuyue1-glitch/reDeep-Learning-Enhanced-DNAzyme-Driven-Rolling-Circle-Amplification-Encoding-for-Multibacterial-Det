from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 128, train: bool = False):
    items = [transforms.Resize((image_size, image_size))]
    if train:
        items.extend([transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)])
    items.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(items)
