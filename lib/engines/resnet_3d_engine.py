from .engine_base import EngineBase

from ..datasets.dataset_3d import Dataset3D
from ..models.resnet_3d import ResNet3D

class ResNet3DEngine(EngineBase):
    def __init__(self, config, tb, logger, **kwargs):
        super().__init__(config, tb, logger, **kwargs)
        self.setup_dataset()
        self.setup_model()

    def train(self):
        num_epochs = self.config.train_epochs
        for epoch in range(num_epochs):
            self.logger.info(
                f"Beginning training epoch {epoch + 1}/{num_epochs}..."
            )
            train_losses = []
            train_labels = []
            train_predictions = []

            for x, y, loss, pred in super().train():
                train_losses.append(loss)
                train_labels.append(y)
                train_predictions.append(pred)

            avg_loss = round(sum(train_losses) / len(train_losses), 7)
            self.logger.info(
                f"Epoch {epoch + 1} training completed!"
                f"\n\t Avg loss: {avg_loss}"
            )

            self.logger.info("Running validation...")
            valid_losses = []
            valid_labels = []
            valid_predictions = []

            for x, y, loss, pred in super().validate():
                valid_losses.append(loss)
                valid_labels.append(y)
                valid_predictions.append(pred)

            avg_loss = round(sum(valid_losses) / len(valid_losses), 7)
            self.logger.info(
                f"Epoch {epoch + 1} validation completed!"
                f"\n\t Avg loss: {avg_loss}"
            )

    def test(self):
        pass

    def setup_dataset(self):
        self.dataset = Dataset3D(self.config, self.logger)

    def setup_model(self):
        num_channels = len(self.config.image_columns)
        self.model = ResNet3D(num_classes=3, num_channels=num_channels)
        self.setup_pretrain_optimizer()
        self.setup_train_optimizer()
