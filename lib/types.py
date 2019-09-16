from typing import NamedTuple, List, Dict, Type

import torch.optim as optim

from lib.engines import Engine, Wang3DEngine


class ImageRecord(NamedTuple):
    patient_id: str
    visit_code: str
    image_paths: List[str]
    label: str


ENGINE_TYPES: Dict[str, Type[Engine]] = {
    "wang_3d": Wang3DEngine
}

OPTIMIZER_TYPES = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop
}
