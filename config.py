import logging
import sys
import uuid
from argparse import Namespace, ArgumentParser
from typing import Tuple, List

import torch
from tensorboardX import SummaryWriter


def _get_config() -> Tuple[Namespace, List[str]]:
    parser = ArgumentParser()

    parser.add_argument("--run-id", type=str, default=f"DEFAULT_{uuid.uuid4().hex.upper()[0:4]}",
                        help="A string that uniquely identifies this run.")
    parser.add_argument("--interactive", dest="interactive", action="store_true", default=False,
                        help="Whether to start the script in interactive mode, where it will initialize the data and"
                             "model but will not perform the training sequence.")
    parser.add_argument("--log-level", type=int, default=10,
                        help="Logging level, see Python logging module for deatils.")
    parser.add_argument("--write-tensorboard", dest="write_tensorboard", action="store_true", default=False,
                        help="Whether to create tensorboard.")
    parser.add_argument("--save-results", dest="save_results", action="store_true", default=False,
                        help="Whether to save the test results for analysis later.")
    parser.add_argument("--log-to-file", dest="log_to_file", action="store_true", default=False,
                        help="Whether to write logs to a file.")
    parser.add_argument("--save-best-model", dest="save_best_model", action="store_true", default=False,
                        help="Whether to save the best model weights (lowest validation loss or highest validation "
                             "accuracy) to be used for testing.")
    parser.add_argument("--log-to-stdout", dest="log_to_stdout", action="store_true", default=False,
                        help="Whether to write logs to a stdout.")
    parser.add_argument("--engine", type=str, default="resnet_3d",
                        help="Which engines to use. See all of the available engines in the lib/types.py.")
    parser.add_argument("--use-gpu", dest="use_gpu", action="store_true", default=False,
                        help="Whether to use GPU for training and testing.")
    parser.add_argument("--remove-outputs-after-completion", dest="remove_outputs_after_completion",
                        action="store_true", default=False, help="Whether to remove outputs (weights, results) after "
                                                                 "the run finishes.")
    parser.add_argument("--dataset-size-limit", type=int, default=-1,
                        help="Limits the size of the data used for experiment, used for debugging purposes. Set to -1 "
                             "to use the whole set.")
    parser.add_argument("--image-column", type=str, default="DX",
                        help="The names of the columns that contain the path to the image.")
    parser.add_argument("--label-column", type=str, default="DX",
                        help="The name of the column that contains the label.")
    parser.add_argument("--brain-mask-path", type=str, default=None,
                        help="Path to the mask_ICV.nii file that is created by Clinica, which is used to strip the "
                             "skull.")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of workers (processes) for the PyTorch dataloader.")
    parser.add_argument("--training-crossval-folds", type=int, default=5,
                        help="K-fold cross-validation for the training dataset.")
    parser.add_argument("--testing-split", type=float, default=0.2,
                        help="Decimal percentage of data allocated to testing.")
    parser.add_argument("--pretrain-epochs", type=int, default=-1,
                        help="Number of pre-training epochs to perform.")
    parser.add_argument("--train-epochs", type=int, default=100,
                        help="Number of training epochs to perform.")
    parser.add_argument("--pretrain-optim-lr", type=float, default=0.001,
                        help="Learning rate for the pretraining optimizer.")
    parser.add_argument("--pretrain-optim-wd", type=float, default=0.01,
                        help="Weight decay for the pretraining optimizer.")
    parser.add_argument("--pretrain-optim-momentum", type=float, default=0.01,
                        help="Momentum for the pretraining optimizer.")
    parser.add_argument("--train-optim-lr", type=float, default=0.001, help="Learning rate for the training optimizer.")
    parser.add_argument("--train-optim-wd", type=float, default=0.01, help="Weight decay for the training optimizer.")
    parser.add_argument("--train-momentum", type=float, default=1.0,
                        help="Momentum for SGD, ignore for Adam optimizer.")
    parser.add_argument("--pretrain-batch-size", type=int, default=2, help="Batch size for pre-training.")
    parser.add_argument("--train-batch-size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--validate-batch-size", type=int, default=2, help="Batch size for validation.")
    parser.add_argument("--test-batch-size", type=int, default=2, help="Batch size for testing.")
    parser.add_argument("--lrate-scheduler", type=str, default="", help="Learning rate scheduler.")
    parser.add_argument("--pretrain-optimizer",
                        type=str, default="adam", help="Optimizer used for pre-training, see engine.py for a list of"
                                                       "added engines or add to the dictionary.")
    parser.add_argument("--train-optimizer",
                        type=str, default="adam", help="Optimizer used for pre-training, see engine.py for a list of"
                                                       "added engines or add to the dictionary.")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of classes in predication: AD, MCI, CN.")
    parser.add_argument("--data-lookup", type=str, default="", help="Path of the file that maps storage location of MRI images with patient information")

    known_config, unknown_config = parser.parse_known_args()

    return known_config, unknown_config


config, unknown = _get_config()


def _get_logger(config: Namespace) -> logging.Logger:
    log_handlers = []

    if config.log_to_stdout:
        log_handlers.append(logging.StreamHandler(sys.stdout))

    if config.log_to_file:
        log_handlers.append(
            logging.FileHandler(filename=f"outputs/logs/{config.run_id}.txt"))
    
    logging.basicConfig(level=config.log_level,
                        handlers=log_handlers,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    logger = logging.getLogger()

    return logger


logger = _get_logger(config)


class SummaryWriterWrapper(object):
    """Wrap around TensorBoard so don't need to wrap functions around if statements.
    """

    def __init__(self, write_tensorboard: bool = False):
        self.write_tensorboard = write_tensorboard

        if self.write_tensorboard:
            self.output_path = f"outputs/tensorboards/{config.run_id}"
            self.summary_writer = SummaryWriter(self.output_path)
        else:
            self.output_path = None
            self.summary_writer = None

    def __getattr__(self, attr):
        if self.write_tensorboard:
            original_attr = self.summary_writer.__getattribute__(attr)

            if callable(original_attr):
                def hooked(*args, **kwargs):
                    result = original_attr(*args, **kwargs)
                    if result == self.summary_writer:
                        return self
                    return result

                return hooked
            else:
                return original_attr
        else:
            return lambda *args, **kwargs: None


tensorboard = SummaryWriterWrapper(config.write_tensorboard)


def configure_device():
    """ Setups the device (CPU vs GPU) for the engine.
    """
    cuda_available = torch.cuda.is_available()
    use_gpu = config.use_gpu
    gpu_count = torch.cuda.device_count()

    if cuda_available and use_gpu and gpu_count > 0:
        logger.info(f"Using {gpu_count} GPU for training.")
        return torch.device("cuda")
    else:
        logger.info(f"Using CPU for training.")
        return torch.device("cpu")


configured_device = configure_device()
