from typing import NamedTuple, Union

import torch.optim as optim


class ImageRecord(NamedTuple):
    patient_id: str
    visit_code: str
    image_path: str
    label: str


Optimizer = Union[optim.Adam, optim.SGD, optim.RMSprop]
