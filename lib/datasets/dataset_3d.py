import torch
import torchvision.transforms as T

from .dataset_base import DatasetBase
from ..transforms import PadToSameDim, NaNToNum, RangeNormalize, MeanStdNormalize

class Dataset3D(DatasetBase):
    def __init__(self, config, logger, **kwargs):
        super().__init__(config, logger, **kwargs)

        self.transforms = T.Compose([
            T.ToTensor(),
            PadToSameDim(3),
            NaNToNum(),
            RangeNormalize(),
            MeanStdNormalize()
        ])

    def __getitem__(self, idx):
        images, label = self.process_item(idx)

        if images is None or None in images:
            return None, None

        transformed_images = [ self.transforms(img) for img in images ]

        if len(transformed_images) == 3:
            stacked_image = torch.stack(transformed_images)
        else:
            stacked_image = images[0]

        return stacked_image, label
