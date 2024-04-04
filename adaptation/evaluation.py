from warnings import filterwarnings

filterwarnings(action='ignore', category=DeprecationWarning)
import argparse
import sys
import os
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
from pre_training.pre_training_algo import PreTrainMultitask
import time
import torch
import robomimic.utils.file_utils as FileUtils

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
from glob import glob

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
    # e.g., experiments/LIBERO_SPATIAL/Multitask/BCRNNPolicy_seed100/
    pre_trained_model_path = '../scripts/experiments/LIBERO_OBJECT/PreTrainMultitask/BCTransformerPolicy_seed10000/run_003/multitask_model.pth'
    adaptor_model_path = '../scripts/experiments/LIBERO_OBJECT/PreTrainMultitask/LoraBCTPolicy_seed10000/run_012/lora_model.pth'

    # load the pre-trained model and adaptor model
    sd, pre_train_cfg, previous_mask = torch_load_model(
        pre_trained_model_path, map_location=None
    )

    # get the cfg
    model_dict = torch.load(adaptor_model_path, map_location=None)
    cfg = model_dict['cfg']
    lora_model_sd = model_dict['state_dict']

    which_bias_train = 'lora_only' if not cfg.adaptation.train_all_bias else 'all'

    control_seed(cfg.seed)
    cfg.folder = get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")

    algo = safe_device(eval('PreTrainMultitask')(10, cfg), 'cuda')
    algo.policy.previous_mask = previous_mask

    algo.policy.load_state_dict(sd, strict=False)
    algo.policy.load_state_dict(lora_model_sd, strict=False)

    ##########################

    if not hasattr(cfg.data, "task_order_index"):
        cfg.data.task_order_index = 0

    # get the benchmark the task belongs to
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    descriptions = [benchmark.get_task(i).language for i in range(10)]
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    # load the dataset via using function get_dataset, so that the obsutils in robomimic can be initialized
    all_obs_keys = []
    for modality_name, modality_list in cfg.data.obs.modality.items():
        all_obs_keys += modality_list

    datasets_default_path = get_libero_path("datasets")
    dataset_path = os.path.join(datasets_default_path, benchmark.get_task_demonstration(0))
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": cfg.data.obs.modality})
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, all_obs_keys=all_obs_keys, verbose=False
    )
    ### ======================= start evaluation ============================
    task_id = [cfg.adaptation.adaptation_task_id]

    # cfg.eval.n_eval=20 if want to set a different eval num as that in the exp
    success_rate = evaluate_success(cfg, algo, benchmark, task_id)
    print(success_rate)


if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    main()
