import sys

sys.path.append("/home/manolotis/sandbox/robustness_benchmark/")
import torch

torch.multiprocessing.set_sharing_strategy('file_system')
from waymo_utils.code.models.data import get_predictions_dataloader
import os
import random
from waymo_utils.code.utils.evaluation import parse_arguments, get_config, averaged_minADE
import numpy as np

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

args = parse_arguments()
evaluation_configs = get_config(args)

for config in evaluation_configs:
    print(f"Evaluating {config['model']['name']}")

    predictions_dataloader = get_predictions_dataloader(config)
    # savefolder = os.path.join(config["output_config"]["out_path"], config["model"]["name"])

    if not os.path.exists(config["output_config"]["out_path"]):
        os.makedirs(config["output_config"]["out_path"], exist_ok=True)

    averaged_minADEs = averaged_minADE(predictions_dataloader)

    np.savez_compressed(
        os.path.join(
            config["output_config"]["out_path"],
            f"{config['model']['name']}.npz"),
        **averaged_minADEs)

    print(f"Saved evaluation to ", config["output_config"]["out_path"])
