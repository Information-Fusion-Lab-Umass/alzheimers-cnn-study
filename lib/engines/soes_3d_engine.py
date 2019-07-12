import torch
from .classification_engine import ClassificationEngine
from ..models.resnet_3d import ResNet3D
from ..models.soes_3d import Soes3D
from ..datasets.dataset_3d import Dataset3D

class Soes3DEngine(ClassificationEngine):
    def __init__(self, config, tb, logger, **kwargs):
        super().__init__(config, tb, logger, **kwargs)
        self.setup_dataset()
        self.setup_model()

    def setup_dataset(self):
        self.dataset = Dataset3D(self.config, self.logger)

    def setup_model(self, from_path=None):
        num_channels = len(self.config.image_columns)

        self.model = Soes3D(num_classes=3, num_channels=num_channels)

        if from_path is not None:
            self.logger.info(f"Loading weights from {from_path}.")
            state_dict = torch.load(from_path)
            self.model.load_state_dict(state_dict)

        self.setup_pretrain_optimizer()
        self.setup_train_optimizer()
