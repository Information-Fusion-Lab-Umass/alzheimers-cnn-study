import sys
from time import time
from typing import Type, Dict

import torch
import yaml

from config import config, unknown, logger
from lib.engines import Engine, WuGoogleNetEngine

# https://github.com/pytorch/pytorch/issues/1485
torch.backends.cudnn.benchmark = True

ENGINE_TYPES: Dict[str, Type[Engine]] = {
    "wu_googlenet": WuGoogleNetEngine
}

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

        if config.pretrain_epochs > -1:
            engine.pretrain(config.pretrain_epochs)

        engine.run()

        end_time = time()

        logger.info(f"Experiment finished in {round(end_time - start_time)} seconds.")
        logger.info(f"----- END ({config.run_id}) -----")
