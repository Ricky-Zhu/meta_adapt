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
    adaptor_model_path = '../scripts/experiments/LIBERO_OBJECT/PreTrainMultitask/LoraBCTPolicy_seed10000/run_009/lora_model.pth'

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
    success_rate = evaluate_success(cfg, algo, benchmark, task_id)
    print(success_rate)
    # start_time = time.time()
    # print('#####################')
    # algo.eval()
    #
    # task = benchmark.get_task(cfg.adaptation.adaptation_task_id)
    # env_args = {
    #     "bddl_file_name": os.path.join(
    #         cfg.bddl_folder, task.problem_folder, task.bddl_file
    #     ),
    #     "camera_heights": cfg.data.img_h,
    #     "camera_widths": cfg.data.img_w,
    # }
    # env_num = 10  # TODO change to 20
    #
    # env = SubprocVectorEnv(
    #     [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
    # )
    # env.reset()
    # env.seed(cfg.seed)
    # algo.reset()
    #
    # init_states_path = os.path.join(
    #     cfg.init_states_folder, task.problem_folder, task.init_states_file
    # )
    # init_states = torch.load(init_states_path)
    # indices = np.arange(env_num) % init_states.shape[0]
    # init_states_ = init_states[indices]
    #
    # dones = [False] * env_num
    # steps = 0
    # obs = env.set_init_state(init_states_)
    # task_emb = benchmark.get_task_emb(cfg.adaptation.adaptation_task_id)
    #
    # num_success = 0
    # for _ in range(5):  # simulate the physics without any actions
    #     env.step(np.zeros((env_num, 7)))
    #
    # with torch.no_grad():
    #     while steps < cfg.eval.max_steps:
    #         steps += 1
    #
    #         data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
    #         actions = algo.policy.get_action(data)
    #         obs, reward, done, info = env.step(actions)
    #
    #         # check whether succeed
    #         for k in range(env_num):
    #             dones[k] = dones[k] or done[k]
    #         if all(dones):
    #             break
    #
    #     for k in range(env_num):
    #         num_success += int(dones[k])
    #
    # success_rate = num_success / env_num
    # env.close()
    # print(f'{success_rate}')


if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    main()
