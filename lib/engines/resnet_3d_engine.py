from .classification_engine import ClassificationEngine
from ..models.resnet_3d import ResNet3D
from ..datasets.dataset_3d import Dataset3D

class ResNet3DEngine(ClassificationEngine):
    def __init__(self, config, tb, logger, **kwargs):
        super().__init__(config, tb, logger, **kwargs)
        self.setup_dataset()
        self.setup_model()

    def setup_dataset(self):
        self.dataset = Dataset3D(self.config, self.logger)

    def setup_model(self):
        num_channels = len(self.config.image_columns)
        self.model = ResNet3D(num_classes=3, num_channels=num_channels)
        self.setup_pretrain_optimizer()
        self.setup_train_optimizer()
