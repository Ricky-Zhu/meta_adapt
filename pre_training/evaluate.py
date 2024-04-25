from warnings import filterwarnings

filterwarnings(action='ignore', category=DeprecationWarning)
import argparse
import sys
import os
from pre_training_algo import PreTrainMultitask
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--experiment_dir", type=str, default="experiments")
    # for which task suite
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"],
        default='libero_object'
    )
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        choices=["bc_rnn_policy", "bc_transformer_policy", "bc_vilt_policy"],
        default='bc_transformer_policy'
    )
    parser.add_argument("--seed", type=int, required=True, default=10000)
    parser.add_argument("--ep", type=int)
    parser.add_argument("--load_task", type=int)
    parser.add_argument("--device_id", type=int)
    parser.add_argument("--save-videos", action="store_true")
    # parser.add_argument('--save_dir',  type=str, required=True)
    args = parser.parse_args()
    args.device_id = "cuda:" + str(args.device_id)
    args.save_dir = f"{args.experiment_dir}_saved"

    # if args.algo == "multitask":
    #     assert args.ep in list(
    #         range(0, 50, 5)
    #     ), "[error] ep should be in [0, 5, ..., 50]"
    # else:
    #     assert args.load_task in list(
    #         range(10)
    #     ), "[error] load_task should be in [0, ..., 9]"
    # return args


def main(seed=None):
    # e.g., experiments/LIBERO_SPATIAL/Multitask/BCRNNPolicy_seed100/

    model_path_folder = '../scripts/experiments/LIBERO_OBJECT/PreTrainMultitask/BCTransformerPolicy_seed10000/run_003'
    files = glob(model_path_folder + '/*.pth')
    files = [os.path.join(model_path_folder, 'multitask_model.pth')]
    for model_path in files:

        sd, cfg, previous_mask = torch_load_model(
            model_path, map_location=None
        )

        use_seed = cfg.seed if seed is None else seed
        control_seed(use_seed)
        cfg.folder = get_libero_path("datasets")
        cfg.bddl_folder = get_libero_path("bddl_files")
        cfg.init_states_folder = get_libero_path("init_states")

        algo = safe_device(eval('PreTrainMultitask')(10, cfg), 'cuda')
        algo.policy.previous_mask = previous_mask

        algo.policy.load_state_dict(sd)

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
        start_time = time.time()
        print('#####################')
        print(f'{model_path.split("/")[-1]}')
        algo.eval()
        for task_id in range(10):
            task = benchmark.get_task(task_id)
            task_emb = benchmark.get_task_emb(task_id)
            env_args = {
                "bddl_file_name": os.path.join(
                    cfg.bddl_folder, task.problem_folder, task.bddl_file
                ),
                "camera_heights": cfg.data.img_h,
                "camera_widths": cfg.data.img_w,
            }

            cfg.eval.n_eval = 50  # iterate over all init conditions
            cfg.eval.use_mp = True
            env_num = min(cfg.eval.num_procs, cfg.eval.n_eval) if cfg.eval.use_mp else 1
            eval_loop_num = (cfg.eval.n_eval + env_num - 1) // env_num

            # Try to handle the frame buffer issue
            env_creation = False

            count = 0
            while not env_creation and count < 5:
                try:
                    if env_num == 1:
                        env = DummyVectorEnv(
                            [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
                        )
                    else:
                        env = SubprocVectorEnv(
                            [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
                        )
                    env_creation = True
                except:
                    time.sleep(5)
                    count += 1
            if count >= 5:
                raise Exception("Failed to create environment")

            ### Evaluation loop
            # get fixed init states to control the experiment randomness
            init_states_path = os.path.join(
                cfg.init_states_folder, task.problem_folder, task.init_states_file
            )
            init_states = torch.load(init_states_path)
            num_success = 0

            for i in range(eval_loop_num):
                env.reset()
                indices = np.arange(i * env_num, (i + 1) * env_num) % init_states.shape[0]
                init_states_ = init_states[indices]

                dones = [False] * env_num
                steps = 0
                algo.reset()
                obs = env.set_init_state(init_states_)

                # dummy actions [env_num, 7] all zeros for initial physics simulation
                dummy = np.zeros((env_num, 7))
                for _ in range(5):
                    obs, _, _, _ = env.step(dummy)

                while steps < cfg.eval.max_steps:
                    steps += 1

                    data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                    actions = algo.policy.get_action(data)

                    obs, reward, done, info = env.step(actions)

                    # check whether succeed
                    for k in range(env_num):
                        dones[k] = dones[k] or done[k]

                    if all(dones):
                        break

                # a new form of success record
                for k in range(env_num):
                    if i * env_num + k < cfg.eval.n_eval:
                        num_success += int(dones[k])

            success_rate = num_success / cfg.eval.n_eval
            env.close()

            print(f'task {task_id}: {success_rate}')

        print('*************************')

        # end_time = time.time()
        # print(f'cost time {end_time - start_time}')


if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    import argparse

    parse = argparse.ArgumentParser()
    parse.add_argument('--seed', type=int, default=100)
    args = parse.parse_args()
    main(args.seed)
