from warnings import filterwarnings
import pickle as pkl

filterwarnings(action='ignore', category=DeprecationWarning)
import argparse
import sys
import os
from pre_training_algo import PreTrainMultitask
import sys
from os.path import dirname, abspath
import multiprocessing
from datetime import datetime

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
from tqdm import tqdm
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
from omegaconf import OmegaConf
from easydict import EasyDict
import yaml


class SimpleLogger():
    def __init__(self, logger_path):
        self.logger_path = logger_path

    def write_and_print(self, sentence, to_print=False):
        with open(self.logger_path, "a") as f:
            f.write(sentence + '\n')
        if to_print:
            print(sentence)


@hydra.main(config_path="../configs", config_name="pre_training", version_base=None)
def main(hydra_cfg):
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    # control seed
    control_seed(cfg.seed)

    # prepare lifelong learning
    cfg.folder = get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")

    N_EVAL = 20  # change this to 5 if want to save obs
    base_path = f'../scripts/experiments/{cfg.benchmark_name}/PreTrainMultitask/{cfg.policy.policy_type}_seed10000/'
    print("use the newest model folder .")
    folders = sorted(glob(os.path.join(base_path, 'run_*')))
    model_path_folder = folders[-1]
    logger_path = os.path.join(model_path_folder, 'evaluate.txt')
    # if os.path.exists(logger_path):
    #     print('there exist evaluation log now remove it')
    #     os.remove(logger_path)
    # if use_newest:
    #     print("use the newest model folder .")
    #     folders = sorted(glob(os.path.join(base_path, 'run_*')))
    #     model_path_folder = folders[-1]
    # else:
    #     assert specific_folder is not None, "specify the model folder!"
    #     model_path_folder = os.path.join(base_path, specific_folder)
    # model_path_folder = '../scripts/experiments/LIBERO_OBJECT/PreTrainMultitask/BCTransformerPolicy_seed10000/run_003'
    logger = SimpleLogger(logger_path=logger_path)
    logger.write_and_print(model_path_folder, to_print=True)
    files = glob(model_path_folder + '/*.pth')

    start_time = datetime.now()

    # Format the date and time
    start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
    logger.write_and_print(f'start time: {start_time}')

    for model_path in files:

        sd, pre_train_cfg, previous_mask = torch_load_model(
            model_path, map_location=None
        )
        cfg.shape_meta = pre_train_cfg.shape_meta
        cfg.experiment_dir = pre_train_cfg.experiment_dir
        algo = safe_device(eval('PreTrainMultitask')(10, cfg), 'cuda')
        algo.policy.previous_mask = previous_mask

        algo.policy.load_state_dict(sd)

        # get the benchmark the task belongs to
        benchmark = get_benchmark(cfg.benchmark_name)(cfg.task_creation.task_order)
        descriptions = [benchmark.get_task(i).language for i in range(benchmark.n_tasks)]
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
        # print('#####################')
        # print(f'{model_path.split("/")[-1]}')
        logger.write_and_print('################')
        logger.write_and_print(f'{model_path.split("/")[-1]}', to_print=True)
        algo.eval()
        avg_suc = []
        for i in range(len(cfg.task_creation.select_tasks)):
            task_id = cfg.task_creation.select_tasks[i]
            task_obs_buffer = []
            task = benchmark.get_task(task_id)
            task_emb = benchmark.get_task_emb(task_id)

            env_args = {
                "bddl_file_name": os.path.join(
                    cfg.bddl_folder, task.problem_folder, task.bddl_file
                ),
                "camera_heights": cfg.data.img_h,
                "camera_widths": cfg.data.img_w,
            }

            cfg.eval.n_eval = N_EVAL  # iterate over all init conditions
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
                    # task_obs_buffer.append(data['obs'])
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
            avg_suc.append(success_rate)
            env.close()
            logger.write_and_print(f'task {task_id} {task.language} :{success_rate}')

            # print(f'task {task_id} {task.language} :{success_rate}')

            # with open(os.path.join(model_path_folder, f'task_{task_id}.pkl'), 'wb') as f:
            #     pkl.dump(task_obs_buffer, f)
            #     f.close()

        # print('*************************')
        logger.write_and_print(f'avg_suc:{np.mean(avg_suc)}', to_print=True)
        logger.write_and_print('#####################')

        # end_time = time.time()
        # print(f'cost time {end_time - start_time}')
    end_time = datetime.now()

    # Format the date and time
    end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
    logger.write_and_print(f'end time: {end_time}')


if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    main()
