from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn

from lib.object import Object


class Module(Object, nn.Module, ABC):
    def __init__(self):
        Object.__init__(self)
        nn.Module.__init__(self)
        ABC.__init__(self)

    @abstractmethod
    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        pass
