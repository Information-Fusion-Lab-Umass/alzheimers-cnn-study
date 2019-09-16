from abc import abstractmethod, ABC
from typing import Optional, Type, Dict

import torch
from torch.optim.optimizer import Optimizer

from lib import Object, OPTIMIZER_TYPES
from lib.datasets import Dataset
from lib.models import Model


class Engine(Object, ABC):
    def __init__(self):
        Object.__init__(self)
        ABC.__init__(self)
        self.model_output_folder = f"outputs/weights/{self.config.run_id}"

        self.device: Optional[torch.device] = None
        self.model: Optional[Type[Model]] = None
        self.dataset: Optional[Type[Dataset]] = None
        self.pretrain_optim: Optional[Type[Optimizer]] = None
        self.train_optim: Optional[Type[Optimizer]] = None

        self.setup_device()
        self.setup_dataset()
        self.setup_model()
        self._setup_optimizer()

    def save_current_model(self, file_name: str):
        output_path = f"{self.model_output_folder}/{file_name}"
        Engine.save_model(self.model, output_path)

    # ==================================================================================================================
    # Abstract methods that need to be implemented in child classes
    # ==================================================================================================================

    @abstractmethod
    def setup_dataset(self):
        pass

    @abstractmethod
    def setup_model(self):
        pass

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
    # Device setup
    # ==================================================================================================================

    def setup_device(self, device: torch.device = None):
        """ Setups the device (CPU vs GPU) for the engine.
        """
        if device is not None:
            self.logger.info(f"Using {device} for training.")
            self.device = device
        else:
            cuda_available = torch.cuda.is_available()
            use_gpu = self.config.use_gpu
            gpu_count = torch.cuda.device_count()

            if cuda_available and use_gpu and gpu_count > 0:
                self.logger.info(f"Using {gpu_count} GPU for training.")
                self.device = torch.device("cuda")
            else:
                self.logger.info(f"Using CPU for training.")
                self.device = torch.device("cpu")

    # ==================================================================================================================
    # Optimizer setup
    # ==================================================================================================================

    def _setup_optimizer(self):
        assert self.model is not None, "Attempting to create optimizer before setting up the model."

        optimizer_params = {
            "lr": self.config.pretrain_optim_lr,
            "weight_decay": self.config.pretrain_optim_wd,
            "momentum": self.config.pretrain_optim_momentum
        }
        optimizer_type = self.config.optimizer

        self.logger.info(f"Building pre-training optimizer with the following parameters: {optimizer_params}")
        self.pretrain_optim = self._build_optimizer(self.model, optimizer_type, optimizer_params)

        optimizer_params = {
            "lr": self.config.train_optim_lr,
            "weight_decay": self.config.train_optim_wd,
            "momentum": self.config.train_momentum
        }
        optimizer_type = self.config.optimizer

        self.logger.info(f"Building training optimizer with the following parameters: {optimizer_params}")
        self.train_optim = self._build_optimizer(self.model, optimizer_type, optimizer_params)

    def _build_optimizer(self,
                         model: Type[Model],
                         optimizer_type: str,
                         optimizer_params: Dict[str, str]) -> Type[Optimizer]:
        optimizer_type = optimizer_type.lower()

        if optimizer_type == "adam":
            del optimizer_params["momentum"]

        return OPTIMIZER_TYPES[optimizer_type](model.parameters(), **optimizer_params)
