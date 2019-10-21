import os
import sys
from shutil import rmtree
from time import time
from typing import Type, Dict

import torch
import yaml

from config import config, unknown, logger, tensorboard
from lib.engines import Engine, WuGoogleNetEngine

# https://github.com/pytorch/pytorch/issues/1485
torch.backends.cudnn.benchmark = True

ENGINE_TYPES: Dict[str, Type[Engine]] = {
    "wu_googlenet": WuGoogleNetEngine, 
    "wang_densenet": WangDenseNetEngine,
    "soes_cnn": SoesCnnEngine,
    "jain_vgg": JainVggEngine
}

if __name__ == "__main__":
    sys.path.append("...")

    engine_type = ENGINE_TYPES[config.engine]
    engine = engine_type()

    try:
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
    except Exception as e:
        raise e from None
    finally:
        if config.remove_outputs_after_completion:
            logger.info("Cleaning up outputs...")

            paths = [
                engine.model_output_folder,
                engine.result_output_folder
            ]

            for path in paths:
                if os.path.exists(path):
                    rmtree(path, ignore_errors=True)
