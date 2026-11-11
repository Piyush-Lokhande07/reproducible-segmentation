import os
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(img_size: int = 256, split: str = "train") -> A.Compose:
    """Returns synchronized spatial and photometric transformations."""
    if split == "train":
        return A.Compose([
            A.Resize(height=img_size, width=img_size, interpolation=1), # cv2.INTER_LINEAR
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size, interpolation=1),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})


class VOCSegmentationContract(Dataset):
    """
    Pascal VOC 2012 Semantic Segmentation Dataset wrapper enforcing
    the data contract (class IDs, ignore label 255, synchronized transforms).
    """
    def __init__(
        self, 
        root: str = "./data", 
        image_set: str = "train", 
        download: bool = True,
        transforms: Optional[A.Compose] = None
    ):
        super().__init__()
        self.image_set = image_set
        self.transforms = transforms
        
        # Load torchvision VOCSegmentation to resolve file indices
        self.voc = datasets.VOCSegmentation(
            root=root,
            year="2012",
            image_set=image_set,
            download=download
        )

    def __len__(self) -> int:
        return len(self.voc)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.voc[idx]
        
        # Convert PIL images to NumPy arrays
        image_np = np.array(image.convert("RGB"), dtype=np.uint8)
        mask_np = np.array(mask, dtype=np.int64) # Contains 0-20 and 255

        if self.transforms is not None:
            # Albumentations handles nearest-neighbor mask interpolation internally
            augmented = self.transforms(image=image_np, mask=mask_np)
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"].long()
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(mask_np).long()

        return image_tensor, mask_tensor


def build_dataloaders(
    data_dir: str = "./data",
    img_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 2,
    download: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """Factory function to build reproducible train and val DataLoaders."""
    train_transforms = get_transforms(img_size=img_size, split="train")
    val_transforms = get_transforms(img_size=img_size, split="val")

    train_dataset = VOCSegmentationContract(
        root=data_dir,
        image_set="train",
        download=download,
        transforms=train_transforms
    )

    val_dataset = VOCSegmentationContract(
        root=data_dir,
        image_set="val",
        download=download,
        transforms=val_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader
