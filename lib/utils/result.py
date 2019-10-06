import pickle
from typing import Dict, Optional

import torch
from torch import Tensor

from lib.object import Object


class Result(Object):
    def __init__(self, scores: Tensor = None, labels: Tensor = None, label_encoder=None):
        super().__init__()

        self.state: Dict[str, Optional[Tensor]] = {
            "scores": None,
            "labels": None,
            "loss": None,
            "label_encoder": label_encoder
        }

        if scores is not None and labels is not None:
            self.append_scores(scores, labels)

    def append_scores(self, scores: Tensor, labels: Tensor) -> None:
        assert scores.shape[0] == labels.shape[0], \
            f"Scores ({scores.shape[0]}) and labels ({labels.shape}) must have the same batch size."

        if self.state["scores"] is None or self.state["labels"]:
            self.state["scores"] = scores
            self.state["labels"] = labels
        else:
            self.state["scores"] = torch.cat([self.state["scores"], scores])
            self.state["labels"] = torch.cat([self.state["labels"], labels])

    def append_loss(self, loss: Tensor):
        if self.state["loss"] is None:
            self.state["loss"] = loss
        else:
            self.state["loss"] = torch.cat([self.state["loss"], loss])

    def calculate_accuracy(self) -> float:
        scores = self.state["scores"]
        labels = self.state["labels"]

        num_correct = (scores.argmax(dim=1) == labels).sum().item()
        num_total = scores.shape[0]

        return (num_correct * 1.0) / num_total

    def calculate_accuracy_by_class(self) -> Dict[str, float]:
        assert self.state["label_encoder"] is not None, "Cannot calculate accuracy by class without a label encoder."
        label_encoder = self.state["label_encoder"]

        return {"la": 0.0}

    def calculate_mean_loss(self) -> Optional[float]:
        if self.state["loss"] is not None:
            return self.state["loss"].mean().item()
        else:
            return None

    def save_state(self, file_path: str):
        if file_path == "":
            self.logger.info("Result file path is emtpy, skipping save state.")
            return

        with open(file_path, "wb") as file:
            pickle.dump(self.state, file)

    @classmethod
    def load_state(cls, file_path: str) -> 'Result':
        with open(file_path, "rb") as file:
            state = pickle.load(file)

        return cls(state)
