from abc import abstractmethod, ABC

import torch.nn as nn

from lib import Object


class Model(Object, nn.Module, ABC):
    def __init__(self):
        Object.__init__(self)
        nn.Module.__init__(self)
        ABC.__init__(self)

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def classification_loss(self):
        pass

    @abstractmethod
    def reconstruction_loss(self):
        pass
