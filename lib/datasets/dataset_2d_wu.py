from lib.datasets.dataset_2d import Dataset2D

from pdb import set_trace

class Dataset2DWu(Dataset2D):
    def __init__(self, config, logger, **kwargs):
        super().__init__(config, logger, **kwargs)

    def __len__(self):
        return super().__len__()
        # TODO: UPDATE THIS

    def __getitem__(self, idx):
        images, label = super().__getitem__(idx)
        set_trace()
        return images, label
