import sys

import matplotlib.pyplot as plt

sys.path.append("/home/manolotis/sandbox/robustness_benchmark/")
import torch

torch.multiprocessing.set_sharing_strategy('file_system')
from waymo_utils.code.models.data import get_predictions_pair_dataloader
import os
import random
from waymo_utils.code.utils.evaluation import parse_arguments, get_config, averaged_minADE, per_example_minADE
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
config = get_config(args)

for model in config["models"]:
    print(f"Evaluating {model['name']} wrt {model['base']}")

    predictions_dataloader = get_predictions_pair_dataloader(config, model)
    # savefolder = os.path.join(config["output_config"]["out_path"], config["model"]["name"])

    if not os.path.exists(config["output_config"]["out_path"]):
        os.makedirs(config["output_config"]["out_path"], exist_ok=True)

    per_example_minADEs = per_example_minADE(predictions_dataloader)

    print(per_example_minADEs["all"]["minADE"].shape)

    # x, y = per_example_minADEs["all"]["minADE"][:,1,-1], per_example_minADEs["all"]["minADE"][:,0,-1]
    # plt.scatter(x,y, alpha=0.2, s=1)
    # plt.hist(y - x, bins=100)
    # plt.show()

    np.savez_compressed(
        os.path.join(
            config["output_config"]["out_path"],
            f"{model['name']}.npz"),
        **per_example_minADEs)

    print(f"Saved evaluation to ", config["output_config"]["out_path"])
