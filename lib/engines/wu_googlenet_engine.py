from pdb import set_trace
from typing import Type, List

import torchvision.transforms as T

from lib.engines import Engine
from lib.models import Model
from lib.models.wu_et_al import GoogleNet
from lib.utils import Result
from lib.utils.transforms import RangeNormalize


class WuGoogleNetEngine(Engine):
    def __init__(self):
        super().__init__()
        set_trace()

    def provide_data_path(self) -> str:
        return "/mnt/nfs/work1/mfiterau/ADNI_data/wu_et_al"

    def provide_model(self) -> Type[Model]:
        return GoogleNet()

    def provide_image_transforms(self) -> List[object]:
        return [
            T.ToTensor(),
            RangeNormalize()
        ]

    def run(self, *inputs, **kwargs) -> None:
        self.train(self.config.train_epochs)

    def train(self, num_epochs: int) -> None:
        train_args = {
            "model": self.model,
            "optimizer": self.train_optim,
            "mapping": self.mapping,
            "reconstruction": False,
            "batch_size": self.config.train_batch_size,
            "num_workers": self.config.num_workers
        }

        for num_epoch in range(num_epochs):
            self.logger.info(f"Staring training for epoch {num_epoch + 1}.")

            train_loop = super().loop_through_data_for_training(**train_args)
            result = Result(label_encoder=self.label_encoder)

            for iter_idx, (_, labels, loss, scores) in enumerate(train_loop):
                result.append_loss(loss)
                result.append_scores(scores, labels)

            self.logger.info(f"Finished training {num_epoch + 1}.")
            self.pretty_print_results(result, "train", num_epoch, label_encoder=self.label_encoder)
