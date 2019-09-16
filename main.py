import sys
import yaml
import torch

from time import time

from lib import ENGINE_TYPES
from config import config, unknown, logger, tensorboard

# https://github.com/pytorch/pytorch/issues/1485
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    sys.path.append("...")

    engine_type = ENGINE_TYPES[config.engine]
    engine = engine_type()

    if config.interactive:
        logger.info("Setup completed, entering interactive mode...")
    else:
        logger.info(f"----- START (Job ID: {config.run_id}) -----\n")
        logger.info(f"Following configurations are used for this run:\n"
                    f"{yaml.dump(vars(config), default_flow_style=False)}"
                    f"Unknown arguments received: {unknown}.")

        start_time = time()

        end_time = time()

        logger.info(f"Experiment finished in {round(end_time - start_time)} seconds.")
        logger.info(f"----- END ({config.run_id}) -----")
