from abc import ABC
from argparse import Namespace
from logging import Logger

from tensorboardX import SummaryWriter

from config import config, logger, tensorboard


class Object(ABC):
    def __init__(self):
        self.config: Namespace = config
        self.logger: Logger = logger
        self.tensorboard: SummaryWriter = tensorboard
