from warnings import filterwarnings
import sys

sys.path.append("..")
filterwarnings(action='ignore', category=DeprecationWarning)
import argparse
import sys
import os
from pre_training.pre_training_algo import PreTrainMultitask
import sys
from os.path import dirname, abspath
import multiprocessing

sys.path.append(dirname(dirname(abspath(__file__))))
# TODO: find a better way for this?
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import hydra
import json
import numpy as np
import pprint
import time
import torch
import robomimic.utils.file_utils as FileUtils
from libero.lifelong.utils import create_experiment_dir
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter
from libero.lifelong.algos import *
from libero.lifelong.datasets import get_dataset, SequenceVLDataset, GroupedTaskDataset
from libero.lifelong.metric import (
    evaluate_loss,
    evaluate_success,
    raw_obs_to_tensor_obs,
)
from libero.lifelong.utils import (
    control_seed,
    safe_device,
    torch_load_model,
    NpEncoder,
    compute_flops,
)
from utils.task_creation import create_tasks
from libero.lifelong.main import get_task_embs

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
from policy import *
import time

benchmark_map = {
    "libero_10": "LIBERO_10",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
}

algo_map = {
    "base": "Sequential",
    "er": "ER",
    "ewc": "EWC",
    "packnet": "PackNet",
    "multitask": "Multitask",
}

policy_map = {
    "bc_rnn_policy": "BCRNNPolicy",
    "bc_transformer_policy": "BCTransformerPolicy",
    "bc_vilt_policy": "BCViLTPolicy",
}


def main():
    # define the pre-trained model path
    model_path = '/home/ruiqi/projects/meta_adapt/scripts/experiments/LIBERO_OBJECT/PreTrainMultitask/BCTransformerPolicy_seed10000/run_002/multitask_model_ep5.pth'

    sd, cfg, previous_mask = torch_load_model(
        model_path, map_location=None
    )

    control_seed(cfg.seed)
    cfg.folder = get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")

    # model with lora definition
    cfg.policy.policy_type = 'LoraBCTPolicy'
    # remove the previous experiment dir
    cfg.pop('experiment_dir')
    algo = safe_device(eval('PreTrainMultitask')(10, cfg), 'cuda')
    algo.policy.previous_mask = previous_mask
    algo.policy.load_state_dict(sd, strict=False)

    lora.mark_only_lora_as_trainable(algo.policy, bias='lora_only')

    #################################
    benchmark = get_benchmark(cfg.task_creation.task_suite)(cfg.task_creation.task_order)
    descriptions = [benchmark.get_task(i).language for i in range(10)]
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    n_manip_tasks = benchmark.n_tasks

    # prepare datasets from the benchmark
    manip_datasets = []
    descriptions = []
    shape_meta = None

    for i in range(n_manip_tasks):
        # currently we assume tasks from same benchmark have the same shape_meta
        try:
            task_i_dataset, shape_meta = get_dataset(
                dataset_path=os.path.join(
                    cfg.folder, benchmark.get_task_demonstration(i)
                ),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=(i == 0),
                seq_len=cfg.data.seq_len,
            )
        except Exception as e:
            print(
                f"[error] failed to load task {i} name {benchmark.get_task_names()[i]}"
            )
            print(f"[error] {e}")
        # add language to the vision dataset, hence we call vl_dataset
        task_description = benchmark.get_task(i).language
        descriptions.append(task_description)
        manip_datasets.append(task_i_dataset)

    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    post_adaptation_dataset = [SequenceVLDataset(manip_datasets[6], task_embs[6])]

    ##################################
    # prepare experiment dir and train
    algo.adapt(post_adaptation_dataset, benchmark, adapt_task_id=6)


if __name__ == "__main__":
    main()
