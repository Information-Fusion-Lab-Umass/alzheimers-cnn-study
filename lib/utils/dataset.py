from abc import ABC
from typing import Tuple, List, Union

import numpy as np
import torch.utils.data as data
import torchvision.transforms as T
from sklearn.preprocessing import LabelEncoder
from torch import Tensor

from lib.object import Object
from lib.utils.collate import invalid_collate
from lib.utils.images import load_nii_image, load_npy_image, load_tiff_image
from lib.utils.mapping import Mapping


class Dataset(Object, ABC, data.Dataset):
    def __init__(self, mapping: Mapping, label_encoder: LabelEncoder):
        Object.__init__(self)
        ABC.__init__(self)
        data.Dataset.__init__(self)
        self.mapping = mapping
        self.label_encoder = label_encoder
        self.transforms = None

    def __len__(self):
        if self.config.dataset_size_limit == -1:
            return len(self.mapping)
        else:
            return self.config.dataset_size_limit

    def __getitem__(self, index: int) -> Tuple[Union[Tensor, np.ndarray], int]:
        if self.transforms is None:
            self.logger.warning("No transforms provided to dataset. "
                                "Call provide_transforms() with a list of transforms.")
        
        record = self.mapping[index]
        image_path = record.image_path
        label = record.label
        
        if image_path[-3:] == "nii" or image_path[-3:] == ".gz":
            image = load_nii_image(image_path)
        elif image_path[-3:] == "npy":
            image = load_npy_image(image_path)
        elif image_path[-4:] == "tiff":
            image = load_tiff_image(image_path)
        else:
            raise Exception(f"Unrecognized file extension: {image_path[-3:]} in {image_path}")
       
        if self.transforms is not None:
            image = self.transforms(image)

        if image_path[-3:] == ".gz":
            image = image.unsqueeze(0)
       
        return image, self.label_encoder.transform([label])

    def provide_transforms(self, transforms: List[object]):
        self.transforms = T.Compose(transforms)

    @classmethod
    def build_loader(cls,
                     mapping: Mapping,
                     label_encoder: LabelEncoder,
                     image_transforms: List[object],
                     **loader_params):
        dataset = cls(mapping, label_encoder)

        default_loader_params = {
            "batch_size": 1,
            "shuffle": True,
            "num_workers": 0,
            "collate_fn": invalid_collate,
            "drop_last": True
        }
        default_loader_params.update(loader_params)

        dataset.provide_transforms(image_transforms)

        return data.DataLoader(dataset, **default_loader_params)
