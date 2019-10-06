import pickle
from pdb import set_trace
from typing import Dict, Optional, Tuple

import torch
from sklearn.preprocessing import LabelEncoder
from torch import Tensor

from lib.object import Object


class Result(Object):
    def __init__(self, scores: Tensor = None, labels: Tensor = None, label_encoder=None):
        super().__init__()

        self.state = {
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

        scores = scores.detach().cpu()
        labels = labels.detach().cpu()

        if self.state["scores"] is None or self.state["labels"] is None:
            self.state["scores"] = scores
            self.state["labels"] = labels
        else:
            self.state["scores"] = torch.cat([self.state["scores"], scores])
            self.state["labels"] = torch.cat([self.state["labels"], labels])

    def append_loss(self, loss: Tensor):
        loss = loss.detach().cpu().unsqueeze(0)

        if self.state["loss"] is None:
            self.state["loss"] = loss
        else:
            self.state["loss"] = torch.cat([self.state["loss"], loss])

    def calculate_accuracy_pct(self) -> float:
        scores = self.state["scores"]
        labels = self.state["labels"]

        return Result._calculate_accuracy(scores, labels)[0]

    def calculate_accuracy_num(self) -> Dict[str, int]:
        scores = self.state["scores"]
        labels = self.state["labels"]
        _, num_correct, num_total = Result._calculate_accuracy(scores, labels)

        return {
            "num_correct": num_correct,
            "num_total": num_total
        }

    def calculate_accuracy_by_class_pct(self) -> Dict[str, float]:
        assert self.state["label_encoder"] is not None, "Cannot calculate accuracy by class without a label encoder."
        label_encoder: LabelEncoder = self.state["label_encoder"]
        classes = label_encoder.classes_
        scores = self.state["scores"]
        labels = self.state["labels"]

        result = {}

        for c in classes:
            class_label = torch.LongTensor(label_encoder.transform([c]))
            target_idx = (labels == class_label).nonzero().squeeze()

            if target_idx.shape == torch.Size([0]):
                continue

            result[c] = Result._calculate_accuracy(scores[target_idx], labels[target_idx])[0]

        return result

    def calculate_accuracy_by_class_num(self) -> Dict[str, int]:
        assert self.state["label_encoder"] is not None, "Cannot calculate accuracy by class without a label encoder."
        label_encoder: LabelEncoder = self.state["label_encoder"]
        classes = label_encoder.classes_
        scores = self.state["scores"]
        labels = self.state["labels"]

        result = {}

        for c in classes:
            class_label = torch.LongTensor(label_encoder.transform([c]))
            target_idx = (labels == class_label).nonzero().squeeze()

            if target_idx.shape == torch.Size([0]):
                continue

            _, num_correct, num_total = Result._calculate_accuracy(scores[target_idx], labels[target_idx])
            result[f"{c}_correct"] = num_correct
            result[f"{c}_total"] = num_total

        return result

    def calculate_mean_loss(self) -> Optional[float]:
        if self.state["loss"] is not None:
            return self.state["loss"].mean().item()
        else:
            return None

    def save_state(self, file_path: str):
        if not self.config.save_results:
            self.logger.info("Configuration save_results set to false, skipping saving results.")
            return

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

    @classmethod
    def _calculate_accuracy(cls, scores: Tensor, labels: Tensor) -> Tuple[float, int, int]:
        try:
            num_correct = (scores.argmax(dim=1) == labels).sum().item()
        except:
            set_trace()
        num_total = len(scores)
        pct_correct = (num_correct * 1.0) / num_total

        return pct_correct, num_correct, num_total
