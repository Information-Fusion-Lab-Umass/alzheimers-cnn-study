import torch
from pdb import set_trace

from .engine_base import EngineBase
from ..datasets.dataset_3d import Dataset3D
from ..result import Result

class ClassificationEngine(EngineBase):
    def __init__(self, config, tb, logger, **kwargs):
        super().__init__(config, tb, logger, **kwargs)

    def train(self):
        num_epochs = self.config.train_epochs
        for epoch in range(num_epochs):
            self.logger.info(
                f"========== Epoch {epoch + 1}/{num_epochs} =========="
            )
            # ==================================================================
            # Training
            # ==================================================================
            train_losses = []
            train_labels = []
            train_predictions = []

            for x, y, loss, pred in super().train():
                train_losses.append(loss)
                train_labels.append(y)
                train_predictions.append(pred)

            avg_loss = round(sum(train_losses) / len(train_losses), 7)
            train_labels = torch.cat(train_labels)
            train_predictions = torch.cat(train_predictions).argmax(dim=1)
            train_result = Result(train_predictions, train_labels)

            self.logger.info(
                f"Epoch {epoch + 1} training completed! Stats:"
                f"\n\t Loss: {avg_loss}"
                f"\n\t Acc: {round(train_result.accuracy(), 4)}"
            )

            # ==================================================================
            # Validation
            # ==================================================================
            self.logger.info("Running validation...")
            valid_losses = []
            valid_labels = []
            valid_predictions = []

            for x, y, loss, pred in super().validate():
                valid_losses.append(loss)
                valid_labels.append(y)
                valid_predictions.append(pred)

            avg_loss = round(sum(valid_losses) / len(valid_losses), 7)
            valid_labels = torch.cat(valid_labels)
            valid_predictions = torch.cat(valid_predictions).argmax(dim=1)
            valid_result = Result(valid_predictions, valid_labels)

            self.logger.info(
                f"Epoch {epoch + 1} validation completed! Stats:"
                f"\n\t Loss: {avg_loss}"
                f"\n\t Acc: {round(valid_result.accuracy(), 4)}"
            )
            self.logger.info(
                f"========== End epoch {epoch + 1}/{num_epochs} =========="
            )

    def test(self):
        labels = []
        predictions = []

        self.logger.info(f"Running test set...")

        for x, y, pred in super().test():
            labels.append(y)
            predictions.append(pred)

        labels = torch.cat(labels)
        predictions = torch.cat(predictions).argmax(dim=1)
        result = Result(predictions, labels)

        self.logger.info(
            f"Test completed, stats:"
            f"\n\t Acc: {round(result.accuracy(), 4)}"
        )
