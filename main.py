import sys
import yaml
import uuid
import torch
import logging

import numpy as np

from time import time
from pdb import set_trace
from argparse import ArgumentParser
from tensorboardX import SummaryWriter

from lib.directory import mkdir

def main(config, tb, logger):
    # https://github.com/pytorch/pytorch/issues/1485
    torch.backends.cudnn.benchmark=True

    if config.engine == "resnet_3d":
        from lib.engines.resnet_3d_engine import ResNet3DEngine as Engine
    elif config.engine == "wang_3d":
        from lib.engines.wang_3d_engine import Wang3DEngine as Engine
    elif config.engine == "soes_3d":
        from lib.engines.soes_3d_engine import Soes3DEngine as Engine
    elif config.engine == "jain_2d":
        from lib.engines.jain_2d_engine import Jain2DEngine as Engine
    else:
        raise Exception(f"Unknown or unsupported engine: {config.engine}.")

    engine = Engine(config, tb, logger)
    k_folds = config.training_crossval_folds
    val_acc = 0.0
    val_acc_list = np.array([0.0] * config.train_epochs)
    for fold_i in range(k_folds):
        acc, acc_list = engine.train(fold_i)
        logger.info("fold " + str(fold_i) + ": " + str(acc))
        val_acc += acc
        val_acc_list = np.add(val_acc_list, np.array(acc_list))
    val_acc /= k_folds
    val_acc_list = np.divide(val_acc_list, k_folds)
    logger.info("Cross-validation Accuracy: " + str(val_acc))
    logger.info(val_acc_list)
 
    result = engine.test()

    if config.save_result:
        result.save_to_file(config.run_id)

def _parse_main_arguments():
    '''Parse the arguments passed to main.py.
    '''
    parser = ArgumentParser()

    parser.add_argument("--run_id",
                        type=str,
                        default=f"DEFAULT_{uuid.uuid4().hex.upper()[0:4]}",
                        help="A string that uniquely identifies this run.")

    parser.add_argument("--log_level",
                        type=int,
                        default=10,
                        help="Logging level, see Python logging module for deatils.")

    parser.add_argument("--write_tensorboard",
                        dest="write_tensorboard",
                        action="store_true",
                        default=False,
                        help="Whether to create tensorboard.")

    parser.add_argument("--save_result",
                        dest="save_result",
                        action="store_true",
                        default=False,
                        help="Whether to save the test results for analysis later.")

    parser.add_argument("--log_to_file",
                        dest="log_to_file",
                        action="store_true",
                        default=False,
                        help="Whether to write logs to a file.")

    parser.add_argument("--save_best_model",
                        dest="save_best_model",
                        action="store_true",
                        default=False,
                        help="Whether to save the best model weights (lowest validation loss or highest validation accuracy) to be used for testing.")

    parser.add_argument("--log_to_stdout",
                        dest="log_to_stdout",
                        action="store_true",
                        default=False,
                        help="Whether to write logs to a stdout.")

    parser.add_argument("--engine",
                        type=str,
                        default="resnet_3d",
                        help="Which engines to use. See all of the available engines in the lib/engines folder. The argument takes the engine class name in snake case and without the word \"engine\", for example, \"resnet_3d\" for ResNet3DEngine ")

    parser.add_argument("--use_gpu",
                        dest="use_gpu",
                        action="store_true",
                        default=False,
                        help="Whether to use GPU for training and testing.")

    parser.add_argument("--dataset_size_limit",
                        type=int,
                        default=-1,
                        help="Limits the size of the data used for experiment, used for debugging purposes. Set to -1 to use the whole set.")

    parser.add_argument("--data_path",
                        type=str,
                        default="data/data_mapping.py",
                        help="Where the mapping file that points to where the MRI data path is located")

    parser.add_argument("--image_columns",
                        type=str,
                        nargs="+",
                        help="The names of the columns that contain the paths to the images. Options are csf_path, gray_matter_path, white_matter_path, skull_intact_path.")

    parser.add_argument("--label_column",
                        type=str,
                        default="DX",
                        help="The name of the column that contains the label.")

    parser.add_argument("--brain_mask_path",
                        type=str,
                        default=None,
                        help="Path to the mask_ICV.nii file that is created by Clinica, which is used to strip the skull.")

    parser.add_argument("--num_workers",
                        type=int,
                        default=8,
                        help="Number of workers (processes) for the PyTorch dataloader.")

    parser.add_argument("--training_crossval_folds",
                        type=int,
                        default=5,
                        help="K-fold cross-validation for the training dataset.")

    parser.add_argument("--testing_split",
                        type=float,
                        default=0.2,
                        help="Decimal percentage of data allocated to testing.")

    parser.add_argument("--pretrain_optim_lr",
                        type=float,
                        default=0.001,
                        help="Learning rate for the pretraining optimizer.")

    parser.add_argument("--pretrain_optim_wd",
                        type=float,
                        default=0.01,
                        help="Weight decay for the pretraining optimizer.")

    parser.add_argument("--pretrain_batch_size",
                        type=int,
                        default=2,
                        help="Batch size for pretraining.")

    parser.add_argument("--train_epochs",
                        type=int,
                        default=100,
                        help="Number of training epochs to perform.")

    parser.add_argument("--train_optim_lr",
                        type=float,
                        default=0.001,
                        help="Learning rate for the training optimizer.")

    parser.add_argument("--train_optim_wd",
                        type=float,
                        default=0.01,
                        help="Weight decay for the training optimizer.")

    parser.add_argument("--train_batch_size",
                        type=int,
                        default=2,
                        help="Batch size for training.")

    parser.add_argument("--train_momentum",
                        type=float,
                        default=1.0,
                        help="Momentum for SGD, ignore for Adam optimizer.")

    parser.add_argument("--validate_batch_size",
                        type=int,
                        default=2,
                        help="Batch size for validation.")

    parser.add_argument("--test_batch_size",
                        type=int,
                        default=2,
                        help="Batch size for testing.")

    parser.add_argument("--lrate_scheduler",
                        type=str,
                        default="",
                        help="Learning rate scheduler.")

    parser.add_argument("--optimizer",
                        type=str,
                        default="Adam",
                        help="Adam or SGD.")

    parser.add_argument("--num_classes",
                        type=int,
                        default=3,
                        help="# classes in predication: AD, MCI, CN.")

    args, unknown = parser.parse_known_args()

    print(f"Unknown arguments received: {unknown}.\n")

    return args

def _get_logger(config):
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

if __name__ == "__main__":
    # Setup PYTHONPATH
    sys.path.append("...")

    config = _parse_main_arguments()
    logger = _get_logger(config)

    if config.write_tensorboard:
        tb = SummaryWriter(log_dir=f"outputs/tensorboards/{config.run_id}.tb")
    else:
        tb = None

    logger.info(f"----- START ({config.run_id}) -----")
    logger.info(
        f"Following configurations are used for this run:\n"
        f"\n{yaml.dump(vars(config), default_flow_style=False)}\n")

    start_time = time()

    main(config, tb, logger)

    logger.info(f"Experiment finished in {round(time() - start_time)} seconds.")
    logger.info(f"----- END ({config.run_id}) -----")

    if config.write_tensorboard:
        tb.close()
