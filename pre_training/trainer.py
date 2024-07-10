from warnings import filterwarnings

filterwarnings(action='ignore', category=DeprecationWarning)
from libero.libero import benchmark
from pre_training_algo import PreTrainMultitask
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import multiprocessing
import pprint
import time
from pathlib import Path

import hydra
import numpy as np
import wandb
import yaml
import torch.nn as nn
from easydict import EasyDict

from omegaconf import OmegaConf

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.algos import get_algo_class, get_algo_list
from libero.lifelong.models import get_policy_list
from libero.lifelong.datasets import GroupedTaskDataset, SequenceVLDataset, get_dataset
from libero.lifelong.metric import evaluate_loss, evaluate_success
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

    ##################### get the dataset (pre_training and adaptation) ####################

    benchmark = get_benchmark(cfg.task_creation.task_suite)(cfg.task_creation.task_order)

    # prepare datasets from the benchmark
    manip_datasets = []
    descriptions = []
    shape_meta = None

    select_tasks = cfg.task_creation.select_tasks

    for i in range(len(select_tasks)):
        # currently we assume tasks from same benchmark have the same shape_meta
        try:
            task_i_dataset, shape_meta = get_dataset(
                dataset_path=os.path.join(
                    cfg.folder, benchmark.get_task_demonstration(select_tasks[i])
                ),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=(i == 0),
                seq_len=cfg.data.seq_len,
            )
        except Exception as e:
            print(
                f"[error] failed to load task {select_tasks[i]} name {benchmark.get_task_names()[select_tasks[i]]}"
            )
            print(f"[error] {e}")
        # add language to the vision dataset, hence we call vl_dataset
        task_description = benchmark.get_task(select_tasks[i]).language
        descriptions.append(task_description)
        manip_datasets.append(task_i_dataset)

    task_embs = get_task_embs(cfg, descriptions)

    # to set the task emb for the benchmark, we need to iterate all the tasks
    all_task_descriptions = []
    for i in range(benchmark.n_tasks):
        all_task_descriptions.append(benchmark.get_task(i).language)
    all_task_embs = get_task_embs(cfg, all_task_descriptions)
    benchmark.set_task_embs(all_task_embs)

    pre_training_dataset = [SequenceVLDataset(ds, emb) for (ds, emb) in
                            zip(manip_datasets, task_embs)]

    print("\n=================== Pretraining ===================")
    print(f" Name: {benchmark.name}")
    print(f" # Tasks: {cfg.task_creation.select_tasks}")

    for i in range(len(cfg.task_creation.select_tasks)):
        print(f"    - Task {cfg.task_creation.select_tasks[i]}:")
        print(f"        {benchmark.get_task(cfg.task_creation.select_tasks[i]).language}")
    print("=======================================================================\n")

    # prepare experiment and update the config
    create_experiment_dir(cfg)
    cfg.shape_meta = shape_meta

    algo = safe_device(get_algo_class(cfg.lifelong.algo)(n_tasks=1, cfg=cfg), cfg.device)

    print(f"[info] start pre training!")

    # save the experiment config file, so we can resume or replay later
    with open(os.path.join(cfg.experiment_dir, "config.json"), "w") as f:
        json.dump(cfg, f, cls=NpEncoder, indent=4)

    algo.train()
    algo.learn_all_tasks(pre_training_dataset, benchmark)
    print(cfg.experiment_dir)
    print("[info] finished learning\n")


if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn'
    # if multiprocessing.get_start_method(allow_none=True) != "fork":
    #     multiprocessing.set_start_method("fork", force=True)
    main()
