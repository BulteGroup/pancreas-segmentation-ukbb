import os
import glob
import random

import numpy as np

from visual import show_element

from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms.compose import MapTransform
from monai.config import KeysCollection
from monai.transforms import (
    AddChanneld,
    Orientationd,
    Compose,
    SpatialPadd,
    LoadNiftid,
    RandRotated,
    RandZoomd,
    NormalizeIntensityd,
    ScaleIntensityd,
    ToTensord,
)


def get_filenames(dataset):
    filenames_list = list()
    for i, filename in enumerate(dataset):
        filenames_list.append(os.path.basename(dataset.data[i]["image"]))
    filenames_list.sort()
    return filenames_list


class Winsorized(MapTransform):
    def __init__(self, keys: KeysCollection):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)
        self.winsorizer = Winsorize()

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.winsorizer(d[key])
        return d


class Winsorize(object):
    def __init__(self, percentile=98):
        self.percentile = percentile

    def __call__(self, img):
        """
        Apply the transform to `img`.
        """
        img[img > np.percentile(img, self.percentile)] = np.percentile(img, self.percentile)
        return img


def get_transforms(end_image_shape):
    train_transforms = Compose(
        [
            LoadNiftid(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Winsorized(keys=["image"]),
            NormalizeIntensityd(keys=["image"]),
            ScaleIntensityd(keys=["image"]),
            SpatialPadd(keys=["image", "label"], spatial_size=end_image_shape),
            # TODO bilinear interpolation in RandRotated and thresholding mask after
            RandRotated(keys=["image", "label"], range_x=3., prob=1, mode="nearest"),
            RandZoomd(keys=["image", "label"], mode="nearest"),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadNiftid(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Winsorized(keys=["image"]),
            NormalizeIntensityd(keys=["image"]),
            ScaleIntensityd(keys=["image"]),
            SpatialPadd(keys=["image", "label"], spatial_size=end_image_shape),
            ToTensord(keys=["image", "label"]),
        ]
    )
    return train_transforms, val_transforms


def create_datasets(root_dir, end_image_shape, batch_size=1, validation_proportion=.1):
    img_dir = os.path.join(root_dir, 'imgs')
    mask_dir = os.path.join(root_dir, 'masks')
    train_images = np.sort(glob.glob(os.path.join(img_dir, '*.nii.gz')))
    train_labels = []
    for img_path in train_images:
        mask_name = os.path.splitext(os.path.splitext(os.path.basename(img_path))[0])[0]+'-seg.nii.gz'
        train_labels.append(os.path.join(mask_dir, mask_name))

    # Create dataset with pytorch dataloader
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]

    # Shuffle files and split into training and validation
    random.seed(10135)
    random.shuffle(data_dicts)
    split_pos = int(np.ceil(len(data_dicts) * validation_proportion))
    train_files, val_files = data_dicts[split_pos:], data_dicts[:split_pos]

    # Loading NIfTI and data augmentation transforms
    train_transforms, val_transforms = get_transforms(end_image_shape)

    # train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    # val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    return train_ds, train_loader, val_ds, val_loader


if __name__ == '__main__':
    from params import params5 as params
    train_ds, train_loader, val_ds, val_loader = create_datasets(root_dir=params['data_folder'],
                                                                 end_image_shape=params['image_shape'],
                                                                 batch_size=params['batch_size'],
                                                                 validation_proportion=params['validation_proportion'])
    show_element(train_ds, number_of_examples=5)
    show_element(val_ds, number_of_examples=5)
