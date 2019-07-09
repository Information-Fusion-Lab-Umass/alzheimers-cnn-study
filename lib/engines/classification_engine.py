import torch
from pdb import set_trace

from .engine_base import EngineBase
from ..datasets.dataset_3d import Dataset3D
from ..result import Result
from ..directory import mkdir

class ClassificationEngine(EngineBase):
    def __init__(self, config, tb, logger, **kwargs):
        super().__init__(config, tb, logger, **kwargs)
        self.weight_folder = f"outputs/weights/{self.config.run_id}"

    def train(self, fold_i):
        num_epochs = self.config.train_epochs
        lowest_validation_loss = float("inf")
        highest_validation_acc = float("-inf")
        
        for epoch in range(num_epochs):
            self.logger.info(
                f"========== Epoch {epoch + 1}/{num_epochs} ==========\n"
            )

            if self.config.lrate_scheduler == "poly":
                #self.train_scheduler.step()
                for param_group in self.train_optim.param_groups:
                    param_group['lr'] = self.config.train_optim_lr * \
                                        (1 - epoch / num_epochs)**0.9
                    print(param_group['lr'])

            for task in ["train", "validate"]:
                self.logger.info(f"Running {task}...")

                losses = []
                labels = []
                predictions = []

                iteration_results = super().train(fold_i) \
                    if task == "train" else super().validate(fold_i)

                for x, y, loss, pred in iteration_results:
                    losses.append(loss)
                    labels.append(y)
                    predictions.append(pred)

                loss = round(sum(losses) / len(losses), 6)
                labels = torch.cat(labels)
                predictions = torch.cat(predictions).argmax(dim=1)
                result = Result(predictions, labels)
                acc = result.accuracy()
                class_acc = result.accuracy_by_class()

                self.logger.info(
                    f"Completed {task}, stats:"
                    f"\n\t Loss: {loss}"
                    f"\n\t Acc: {round(acc, 4)}"
                )

                if task == "validate":
                    if acc > highest_validation_acc:
                        highest_validation_acc = acc
                    if  self.config.save_best_model:
                        mkdir(self.weight_folder)

                        if acc > highest_validation_acc:
                            self.logger.info(
                                "Highest validation accuracy! Saving...")
                            self.save_current_model(
                                f"{self.weight_folder}/{highest_validation_acc}.pt")
                        if loss < lowest_validation_loss:
                            self.logger.info(
                                "Lowest validation loss! Saving...")
                            self.save_current_model(
                                f"{self.weight_folder}/{lowest_validation_loss}.pt")

            self.logger.info(
                f"========== End epoch {epoch + 1}/{num_epochs} ==========\n"
            )

        return highest_validation_acc

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
