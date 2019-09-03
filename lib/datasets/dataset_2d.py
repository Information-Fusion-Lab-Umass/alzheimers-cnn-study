from .dataset_base import DatasetBase


class Dataset2D(DatasetBase):
    def __init__(self, config, logger, **kwargs):
        super().__init__(config, logger, **kwargs)

    def __getitem__(self, idx):
        images, label = self.process_item(idx)

        return images, label

    def process_item(self, idx):
        images, label = super().process_item(idx)

        return images, label
