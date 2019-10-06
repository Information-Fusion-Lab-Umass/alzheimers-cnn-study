from abc import abstractmethod, ABC
from typing import Tuple

from torch import Tensor

from lib import Object
from lib.models.module import Module


class Model(Module, ABC):
    def __init__(self):
        Object.__init__(self)
        Module.__init__(self)
        ABC.__init__(self)

    @abstractmethod
    def forward(self, images: Tensor) -> Tensor:
        pass

    @abstractmethod
    def classification_loss(self, images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """Should return a tuple of two Tensors, the first being the loss, the second being the predicted output.
        """
        pass

    def reconstruction_loss(self, images: Tensor) -> Tensor:
        pass
