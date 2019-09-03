import torch

from .classification_engine import ClassificationEngine
from ..datasets.dataset_3d import Dataset3D
from ..models.wang_3d import Wang3D


class Wang3DEngine(ClassificationEngine):
    def __init__(self, config, tb, logger, **kwargs):
        super().__init__(config, tb, logger, **kwargs)
        self.setup_dataset()
        self.setup_model()

    def setup_dataset(self):
        self.dataset = Dataset3D(self.config, self.logger)

    def setup_model(self, from_path=None):
        num_channels = len(self.config.image_columns)

        self.model = Wang3D(num_classes=3, num_channels=num_channels)

        if from_path is not None:
            self.logger.info(f"Loading weights from {from_path}.")
            state_dict = torch.load(from_path)
            self.model.load_state_dict(state_dict)

        self.setup_pretrain_optimizer()
        self.setup_train_optimizer()
