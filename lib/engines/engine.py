import os
from abc import abstractmethod, ABC
from typing import Optional, Type, Dict, Union, List, Iterable, Tuple

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
    def __init__(self):
        Object.__init__(self)
        ABC.__init__(self)
        self.model_output_folder = f"outputs/weights/{self.config.run_id}"
        self.result_output_folder = f"outputs/results/{self.config.run_id}"

        if self.config.save_best_model and not os.path.exists(self.model_output_folder):
            os.mkdir(self.model_output_folder)
        if self.config.save_results and not os.path.exists(self.result_output_folder):
            os.mkdir(self.result_output_folder)

        self.model: Optional[Union[Type[Model], nn.DataParallel]] = None
        self.optimizer: Optional[Type[Optimizer]] = None

        self.logger.info(f"Loading data mapping...")
        manifest_file = f"{self.provide_data_path()}/manifest.csv"
        self.mapping: Mapping = Mapping(mapping_path=manifest_file)

        self.logger.info(f"Building label encoder...")
        labels = list(map(lambda x: x.label, self.mapping))
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
        self.logger.info(f"Created encoder with classes: {self.label_encoder.classes_}.")

    @property
    def num_gpus(self) -> int:
        return torch.cuda.device_count()

    @property
    def device(self) -> torch.device:
        return configured_device

    def save_current_model(self, file_name: str):
        if not self.config.save_best_model:
            self.logger.info("Configuration save_best_model set to false, skipping checkpoint.")
            return

        output_path = f"{self.model_output_folder}/{file_name}"
        Engine.save_model(self.model, output_path)
        self.logger.info(f"Saved current model to {output_path}")

    def pretty_print_results(self,
                             result: Result,
                             step: str,
                             name: str,
                             epoch: int = 0) -> None:
        """Prints the Result object in a human-friendly format as well as logging the results to the session tensorboard
        instance.

        Args:
            result: The lib.utils.result.Result object.
            step: e.g. "train", "validation", or "test"
            name: e.g. "1st fold", "
            epoch: The nth training epoch
        """
        overall_accuracy_pct = {"accuracy": result.calculate_accuracy_pct()}
        overall_accuracy_num = result.calculate_accuracy_num()
        class_accuracy_pct = result.calculate_accuracy_by_class_pct()
        class_accuracy_num = result.calculate_accuracy_by_class_num()
        loss = result.calculate_mean_loss()

        self.logger.info(
            f"Epoch {epoch + 1} {name} results:"
            f"\n\t mean loss: {loss}"
            f"\n\t overall accuracy pct: {overall_accuracy_pct}"
            f"\n\t overall accuracy: {overall_accuracy_num}"
            f"\n\t class accuracy pct: {class_accuracy_pct}"
            f"\n\t class accuracy: {class_accuracy_num}"
        )

        # space is illegal in Tensorboard
        name = "_".join(name.split(" "))

        if loss is not None:
            self.tensorboard.add_scalar(f"{step}/{name}/loss", loss, epoch)

        self.tensorboard.add_scalars(f"{step}/{name}/pct_accuracy", {
            **overall_accuracy_pct, **class_accuracy_pct
        }, epoch)

        self.tensorboard.add_scalars(f"{step}/{name}/num_accuracy", {
            **overall_accuracy_num, **class_accuracy_num
        }, epoch)

    # ==================================================================================================================
    # Abstract methods that need to be implemented in child classes
    # ==================================================================================================================

    @abstractmethod
    def provide_model(self) -> Model:
        """Inherit this method to return an instantiated model to be used.
        """
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
        expects at minimum the T.ToTensor() transform.
        """
        return [T.ToTensor()]

    @abstractmethod
    def run(self, *inputs, **kwargs) -> None:
        """Override this method to define the run logic for the engine.
        """
        pass

    # ==================================================================================================================
    # Training and testing code
    # ==================================================================================================================
    def get_training_args(self, **kwargs) -> Dict[str, object]:
        """Default arguments to be passed into loop_through_data_for_training.
        """
        default_args = {
            "model": self.model,
            "optimizer": self.optimizer,
            "mapping": self.mapping,
            "reconstruction": False,
            "batch_size": self.config.train_batch_size,
            "num_workers": self.config.num_workers
        }

        default_args.update(kwargs)

        return default_args

    def loop_through_data_for_training(self,
                                       model: Union[Model, nn.DataParallel],
                                       optimizer: Optimizer,
                                       mapping: Mapping,
                                       reconstruction: bool = False,
                                       **loader_params):
        """A generator function that takes in a model, optimizer, and data mapping, and loops through the data with the
        model.

        Example:
            default_args = self.get_training_args(model=different_model, reconstruction=true)
            for (input_images, labels, loss, predictions) in loop_through_data_for_training(**default_args):
                # do stuff with loss or predictions

        Args:
            model: The model to train the data with.
            optimizer: The optimizer with the model's parameters set as target.
            mapping: A lib.util.mapping.Mapping object.
            reconstruction: Whether to calculate the reconstruction loss instead of the classification loss.
            **loader_params: Any key-value arguments applicable to the torch.utils.data.DataLoader class, see a list
                here https://pytorch.org/docs/stable/data.html
        """
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

    def get_testing_args(self, **kwargs) -> Dict[str, object]:
        """Default arguments to be passed into loop_through_data_for_training.
        """
        default_args = {
            "model": self.model,
            "mapping": self.mapping,
            "batch_size": self.config.train_batch_size,
            "num_workers": self.config.num_workers
        }

        default_args.update(kwargs)

        return default_args

    @torch.no_grad()
    def loop_through_data_for_testing(self,
                                      model: Union[Model, nn.DataParallel],
                                      mapping: Mapping,
                                      **loader_params) -> Result:
        """A function that loops through the provided data and returns a lib.utils.result.Result object, which can then
        be used to calculate scores.

        Args:
            model: The model to make predictions.
            mapping: A lib.util.mapping.Mapping object.
            **loader_params: Any key-value arguments applicable to the torch.utils.data.DataLoader class, see a list
                here https://pytorch.org/docs/stable/data.html

        Returns: A lib.utils.result.Result object.
        """
        loader = Dataset.build_loader(mapping,
                                      label_encoder=self.label_encoder,
                                      image_transforms=self.provide_image_transforms(),
                                      **loader_params)

        result = Result(label_encoder=self.label_encoder)

        for iter_idx, (images, labels) in enumerate(loader):
            model.eval()
            model.to(device=self.device)

            images = images.float().to(device=self.device)
            labels = labels.squeeze().long().to(device=self.device)

            if type(model) == torch.nn.DataParallel:
                loss, scores = model.module.classification_loss(images, labels)
            else:
                loss, scores = model.classification_loss(images, labels)

            result.append_scores(scores, labels)
            result.append_loss(loss)

        return result

    # ==================================================================================================================
    # Class methods
    # ==================================================================================================================

    @classmethod
    def save_model(cls, model: Union[Model, torch.nn.DataParallel], output_path: str):
        model = model.cpu()

        if type(model) == torch.nn.DataParallel:
            torch.save(model.module.state_dict(), output_path)
        else:
            torch.save(model.state_dict(), output_path)

    # ==================================================================================================================
    # Model and optimizer setup
    # ==================================================================================================================

    def get_optimizer_args(self, **kwargs):
        default_args = {
            "lr": self.config.train_optim_lr,
            "weight_decay": self.config.train_optim_wd,
            "momentum": self.config.train_momentum
        }

        default_args.update(kwargs)

        return default_args

    def build_model(self, from_path: str = None, **optimizer_params) -> Tuple[Type[Model], Optional[Optimizer]]:
        """Build a model and optimizer from the given params. If from_path is provided, weights from from_path will
        be used to initialize the model.
        """
        model = self.provide_model()
        optimizer = None

        if from_path is not None:
            model = Engine.initialize_model(model, from_path)

        if len(optimizer_params.items()) > 0:
            optimizer_type = optimizer_params.get("optimizer_type", "sgd")
            del optimizer_params["optimizer_type"]
            optimizer = Engine.build_optimizer(model.parameters(), optimizer_type, **optimizer_params)

        return model, optimizer

    @classmethod
    def initialize_model(cls, model: Type[Model], from_path: str) -> Model:
        """Initialize the weights of the model from file containing previously saved weights.

        Args:
            model: Model whose weights will be initialized from file.
            from_path: Path to the file.

        Returns: A model with weights initialized from file containing previously saved weights.
        """
        state_dict = torch.load(from_path)
        model.load_state_dict(state_dict)
        return model

    OPTIMIZER_TYPES = {
        "adam": optim.Adam,
        "sgd": optim.SGD,
        "rmsprop": optim.RMSprop
    }

    @classmethod
    def build_optimizer(cls,
                        parameters: Iterable[object],
                        optimizer_type: str,
                        **optimizer_params) -> Optimizer:
        optimizer_type = optimizer_type.lower()

        if optimizer_type == "adam":
            del optimizer_params["momentum"]

        return Engine.OPTIMIZER_TYPES[optimizer_type](parameters, **optimizer_params)
