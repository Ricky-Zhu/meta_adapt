from warnings import filterwarnings

filterwarnings(action='ignore', category=DeprecationWarning)
import os
from pre_training_algo import PreTrainMultitask

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import multiprocessing
import pprint
import hydra
import wandb
import yaml
from easydict import EasyDict
from omegaconf import OmegaConf
from libero.libero import get_libero_path
from libero.lifelong.algos import get_algo_class, get_algo_list

from libero.lifelong.utils import (
    NpEncoder,
    compute_flops,
    control_seed,
    safe_device,
    torch_load_model,
    create_experiment_dir,
    get_task_embs,
)
import sys
from os.path import dirname, abspath

sys.path.append(dirname(dirname(abspath(__file__))))
from utils.task_creation import create_tasks


@hydra.main(config_path="../configs", config_name="pre_training", version_base=None)
def main(hydra_cfg):
    # preprocessing
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    # print configs to terminal
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(cfg)

    # control seed
    control_seed(cfg.seed)

    # prepare lifelong learning
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")

    pre_training_dataset, post_adaptation_dataset, benchmark_instance, shape_meta = create_tasks(cfg)

    print("\n=================== Pretraining ===================")
    print(f" Name: {benchmark_instance.name}")
    print(f" # Tasks: {cfg.task_creation.pre_training_num}")

    for i in range(cfg.task_creation.pre_training_num):
        print(f"    - Task {i + 1}:")
        print(f"        {benchmark_instance.get_task(i).language}")
    print("=======================================================================\n")

    # prepare experiment and update the config
    create_experiment_dir(cfg)
    cfg.shape_meta = shape_meta

    if cfg.use_wandb:
        wandb.init(project="libero_pretraining", config=cfg)
        wandb.run.name = cfg.experiment_name

    algo = safe_device(get_algo_class(cfg.lifelong.algo)(n_tasks=1, cfg=cfg), cfg.device)

    print(f"[info] start lifelong learning with algo {cfg.lifelong.algo}")
    # GFLOPs, MParams = compute_flops(algo, pre_training_dataset[0], cfg)
    # print(f"[info] policy has {GFLOPs:.1f} GFLOPs and {MParams:.1f} MParams\n")

    # save the experiment config file, so we can resume or replay later
    with open(os.path.join(cfg.experiment_dir, "config.json"), "w") as f:
        json.dump(cfg, f, cls=NpEncoder, indent=4)

    algo.train()
    s_fwd, l_fwd = algo.learn_all_tasks(pre_training_dataset, benchmark_instance)

    print("[info] finished learning\n")
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn'
    # if multiprocessing.get_start_method(allow_none=True) != "spawn":
    #     multiprocessing.set_start_method("spawn", force=True)
    main()
