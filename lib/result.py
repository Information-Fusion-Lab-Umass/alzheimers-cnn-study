import torch
import pickle

from pdb import set_trace

class Result(object):
    def __init__(self, predictions, labels):
        assert len(predictions) == len(labels), "The length of the predictions and the length of the labels do not match!"
        self.predictions = predictions
        self.labels = labels

    def accuracy(self):
        num_correct = (self.predictions == self.labels).sum().item()
        total = len(self.predictions)

        return (num_correct * 1.0) / total

    def accuracy_by_class(self):
        classes = torch.cat([self.predictions, self.labels]).unique()
        acc = {}

        for c in classes:
            targets = self.labels == c
            num_correct = (self.predictions == self.labels).sum().item()
            if targets.sum().item() < 0.1:
                acc[c] = 0.0
            else:
                acc[c] =  num_correct * 1.0 / targets.sum().item()

        return acc

    def save_to_file(self, file_name):
        state = {
            "predictions": self.predictions,
            "labels": self.labels
        }

        with open(f"{file_name}.result", "wb") as file:
            pickle.dump(state, file)

    @classmethod
    def load_from_file(cls, file_name):
        with open(f"{file_name}.result", "wb") as file:
            state = pickle.load(file)

        return cls(state["predictions"], state["labels"])
