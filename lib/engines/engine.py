from abc import abstractmethod, ABC
from typing import Optional, Type, Dict, Union, List

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from sklearn.preprocessing import LabelEncoder

from config import configured_device
from lib import Object
from lib.models import Model
from lib.types import Optimizer
from lib.utils import Mapping, Dataset, Result


class Engine(Object, ABC):
    OPTIMIZER_TYPES = {
        "adam": optim.Adam,
        "sgd": optim.SGD,
        "rmsprop": optim.RMSprop
    }

    def __init__(self):
        Object.__init__(self)
        ABC.__init__(self)
        self.model_output_folder = f"outputs/weights/{self.config.run_id}"
        self.result_output_folder = f"outputs/results/{self.config.run_id}"

        self.pretrain_optim: Optional[Type[Optimizer]] = None
        self.train_optim: Optional[Type[Optimizer]] = None

        self.device = configured_device

        self.logger.info(f"Loading data mapping...")
        manifest_file = f"{self.provide_data_path()}/manifest.csv"
        self.mapping: Mapping = Mapping(mapping_path=manifest_file)

        self.logger.info(f"Building label encoder...")
        labels = list(map(lambda x: x.label, self.mapping))
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
        self.logger.info(f"Created encoder with classes: {self.label_encoder.classes_}.")

        self.logger.info(f"Setting up model...")
        self.model: Union[Type[Model], nn.DataParallel] = self.provide_model()

        self.logger.info(f"Setting up optimizer...")
        self.setup_optimizer()

    def save_current_model(self, file_name: str):
        output_path = f"{self.model_output_folder}/{file_name}"
        Engine.save_model(self.model, output_path)
        self.logger.info(f"Saved current model to {output_path}")

    def pretty_print_results(self,
                             result: Result,
                             name: str,
                             epoch: int = 0,
                             label_encoder: LabelEncoder = None):
        overall_accuracy = {"accuracy": result.calculate_accuracy()}
        class_accuracy = result.calculate_accuracy_by_class()
        loss = result.calculate_mean_loss()

        self.logger.info(
            f"Epoch {epoch + 1} {name} results:"
            f"\n\t mean loss: {loss}"
            f"\n\t overall accuracy: {overall_accuracy}"
            f"\n\t class accuracy: {class_accuracy}"
        )

        if loss is not None:
            self.tensorboard.add_scalar(f"{name}/loss", loss, epoch)

        self.tensorboard.add_scalars(f"{name}/accuracy", {
            **overall_accuracy, **class_accuracy
        }, epoch)

    # ==================================================================================================================
    # Abstract methods that need to be implemented in child classes
    # ==================================================================================================================

    @abstractmethod
    def provide_model(self):
        pass

    @abstractmethod
    def provide_data_path(self) -> str:
        """Inherit this method to provide a path containing the images and the manifest.csv file. Engine will build a
        mapping file and save it to self.mapping based on this path.
        """
        pass

    @abstractmethod
    def provide_image_transforms(self) -> List[object]:
        """Override this method to provide a list of image transforms that will be applied to each image. The method
        expects at the very least the T.ToTensor() transform.
        """
        return [T.ToTensor()]

    def pretrain(self, num_epochs: int) -> None:
        pass

    @abstractmethod
    def run(self, *inputs, **kwargs) -> None:
        """Override this method to define the run logic for the engine.
        """
        pass

    # ==================================================================================================================
    # Training and testing code
    # ==================================================================================================================
    def loop_through_data_for_training(self,
                                       model: Union[Model, nn.DataParallel],
                                       optimizer: Optimizer,
                                       mapping: Mapping,
                                       reconstruction: bool = False,
                                       **loader_params):
        loader = Dataset.build_loader(mapping,
                                      label_encoder=self.label_encoder,
                                      image_transforms=self.provide_image_transforms(),
                                      **loader_params)

        for iter_idx, (images, labels) in enumerate(loader):
            model.to(device=self.device)
            images = images.float().to(device=self.device)

            optimizer.zero_grad()

            pred = None

            if reconstruction:
                if type(model) == torch.nn.DataParallel:
                    loss = model.module.reconstruction_loss(images)
                else:
                    loss = model.reconstruction_loss(images)
            else:
                labels = labels.squeeze().long().to(device=self.device)

                if type(model) == torch.nn.DataParallel:
                    loss, pred = model.module.classification_loss(images, labels)
                else:
                    loss, pred = model.classification_loss(images, labels)

            loss.backward()
            optimizer.step()

            yield images.detach().cpu(), \
                  labels.detach().cpu(), \
                  loss.detach().cpu(), \
                  pred.detach().cpu() if pred is not None else pred

    @torch.no_grad()
    def loop_through_data_for_testing(self,
                                      model: Union[Model, nn.DataParallel],
                                      mapping: Mapping,
                                      **loader_params) -> Result:
        loader = Dataset.build_loader(mapping,
                                      label_encoder=self.label_encoder,
                                      image_transforms=self.provide_image_transforms(),
                                      **loader_params)

        result = Result(label_encoder=self.label_encoder)

        for iter_idx, (images, labels) in enumerate(loader):
            model.eval()
            model.to(device=self.device)

            images = images.float().to(device=self.device)
            labels = labels.long().to(device=self.device)

            if type(model) == torch.nn.DataParallel:
                scores = model.module(images)
            else:
                scores = model(images)

            result.append_scores(scores, labels)

        return result

    # ==================================================================================================================
    # Class methods
    # ==================================================================================================================

    @classmethod
    def save_model(cls, model: Type[Model], output_path: str):
        model = model.cpu()

        if type(model) == torch.nn.DataParallel:
            torch.save(model.module.state_dict(), output_path)
        else:
            torch.save(model.state_dict(), output_path)

    # ==================================================================================================================
    # Optimizer setup
    # ==================================================================================================================

    def setup_optimizer(self):
        assert self.model is not None, "Attempting to create optimizer before setting up the model."

        optimizer_params = {
            "lr": self.config.pretrain_optim_lr,
            "weight_decay": self.config.pretrain_optim_wd,
            "momentum": self.config.pretrain_optim_momentum
        }
        optimizer_type = self.config.pretrain_optimizer

        self.logger.info(f"Building pre-training optimizer with the following parameters: {optimizer_params}")
        self.pretrain_optim = Engine._build_optimizer(self.model, optimizer_type, optimizer_params)

        optimizer_params = {
            "lr": self.config.train_optim_lr,
            "weight_decay": self.config.train_optim_wd,
            "momentum": self.config.train_momentum
        }
        optimizer_type = self.config.train_optimizer

        self.logger.info(f"Building training optimizer with the following parameters: {optimizer_params}")
        self.train_optim = Engine._build_optimizer(self.model, optimizer_type, optimizer_params)

    @classmethod
    def _build_optimizer(cls,
                         model: Type[Model],
                         optimizer_type: str,
                         optimizer_params: Dict[str, str]) -> Type[Optimizer]:
        optimizer_type = optimizer_type.lower()

        if optimizer_type == "adam":
            del optimizer_params["momentum"]

        return Engine.OPTIMIZER_TYPES[optimizer_type](model.parameters(), **optimizer_params)
