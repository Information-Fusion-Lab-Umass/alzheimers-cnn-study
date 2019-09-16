from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import torch.utils.data as torch_data
import torchvision.transforms as T
from torch import Tensor

from lib import Object, ImageRecord
from lib.datasets import Mapping
from lib.utils.collate import invalid_collate
from lib.utils.images import load_image
from lib.utils.labels import encode_label


class Dataset(Object, ABC, torch_data.Dataset):
    def __init__(self, mapping: Mapping):
        Object.__init__(self)
        ABC.__init__(self)
        self.mapping = mapping
        self.transforms = T.Compose(self.provide_transforms())

    def __len__(self):
        return len(self.mapping) * self.num_slices_per_image()

    # ==================================================================================================================
    # Abstract methods - OVERRIDE THESE
    # ==================================================================================================================

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        pass

    @abstractmethod
    def provide_transforms(self) -> List:
        """Override this method to specify transforms that will be applied to the images.
        """
        return []

    # ==================================================================================================================
    # Properties, override as needed
    # ==================================================================================================================

    @classmethod
    def num_slices_per_image(cls) -> int:
        """How many slices to count per image, use 1 for 3D images.
        WARN: DO NOT CHANGE THIS VALUE HERE, OVERRIDE IN CHILD CLASSES
        """
        return 1

    @classmethod
    def after_loading_image(cls, image: np.ndarray) -> np.ndarray:
        """Override in child class to add postprocess logic after loading an image.
        """
        return image

    def process_record(self, record: ImageRecord) -> Tuple[List[np.ndarray], int]:
        paths = record.image_paths
        label = record.label

        images = []

        for path in paths:
            image = load_image(path)
            post_processed_image = self.after_loading_image(image)
            images.append(post_processed_image)

        return images, encode_label(label)

    @classmethod
    def build(cls, mapping, **loader_params):
        dataset = cls(mapping)

        default_loader_params = {
            "batch_size": 1,
            "shuffle": True,
            "num_workers": 0,
            "collate_fn": invalid_collate
        }
        default_loader_params.update(loader_params)

        return torch_data.DataLoader(dataset, **default_loader_params)
