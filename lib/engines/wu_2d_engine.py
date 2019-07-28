from lib.engines.classification_engine import ClassificationEngine
from torchvision.models import alexnet
from lib.datasets.dataset_2d_wu import Dataset2DWu

class Wu2DCaffeEngine(ClassificationEngine):
    def __init__(self, config, tb, logger, **kwargs):
        super().__init__(config, tb, logger, **kwargs)
        self.setup_dataset()
        self.setup_model()

    def setup_model(self):
        self.model = alexnet(pretrained=True)
        self.setup_train_optimizer()

    def setup_dataset(self):
        self.dataset = Dataset2DWu(self.config, self.logger)
