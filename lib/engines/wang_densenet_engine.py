from typing import List

import torchvision.transforms as T

from lib.engines import Engine
from lib.models import Model
from lib.models.wang_et_al import DenseNet
from lib.utils import Result
from lib.utils.mapping import Mapping
from lib.utils.transforms import RangeNormalize


class WangDenseNetEngine(Engine):
    def provide_data_path(self) -> str:
        return "/mnt/nfs/work1/mfiterau/ADNI_data/wang_et_al"

    def provide_model(self) -> Model:
        return DenseNet()

    def provide_image_transforms(self) -> List[object]:
        return [
            T.ToTensor(),
            RangeNormalize()
        ]

    def run(self, *inputs, **kwargs) -> None:
        pass

    def train(self,
              num_epochs: int,
              ith_fold: int,
              train_mapping: Mapping,
              valid_mapping: Mapping) -> None:
        pass

    def test(self, ith_fold: int, test_mapping: Mapping) -> Result:
        pass
