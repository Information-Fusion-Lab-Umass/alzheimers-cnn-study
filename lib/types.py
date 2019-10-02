from typing import NamedTuple, Dict, Type

import torch.optim as optim

from lib.engines import Engine, Wu2DEngine


class ImageRecord(NamedTuple):
    patient_id: str
    visit_code: str
	image_path: str
    label: str


ENGINE_TYPES: Dict[str, Type[Engine]] = {
	"wu_2d": Wu2DEngine
}

OPTIMIZER_TYPES = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop
}
