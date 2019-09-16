from typing import List, Tuple

import torch
import torchvision.transforms as T
from torch import Tensor

from lib.datasets import Dataset, Mapping
from lib.utils.transforms import PadToSameDim, NaNToNum, RangeNormalize, MeanStdNormalize


class WangDataset(Dataset):
    def __init__(self, mapping: Mapping):
        super().__init__(mapping)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        image_record = self.mapping[index]
        images, label = super().process_record(image_record)

        transformed_images = [self.transforms(image) for image in images]

        if len(transformed_images) == 3:
            stacked_image = torch.stack(transformed_images)
        else:
            stacked_image = transformed_images[0].unsqueeze(0)

        return stacked_image, label

    def provide_transforms(self) -> List:
        return [
            T.ToTensor(),
            PadToSameDim(3),
            NaNToNum(),
            RangeNormalize(),
            MeanStdNormalize()
        ]
