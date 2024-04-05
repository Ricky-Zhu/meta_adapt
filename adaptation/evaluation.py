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
import robomimic.utils.file_utils as FileUtils
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.algos import *

from libero.lifelong.utils import (
    control_seed,
    safe_device,
    torch_load_model,
    NpEncoder,
    compute_flops,
)
from utils.task_creation import create_tasks
from libero.lifelong.main import get_task_embs
from utils.evaluation import *
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
from policy import *
from glob import glob


def main():
    def evaluate_one_repo_adaptor(pre_train_model_path, adaptor_model_path):
        # load the pre-trained model and adaptor model
        sd, pre_train_cfg, previous_mask = torch_load_model(
            pre_train_model_path, map_location=None
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
        success_rate = evaluate_success_all_init_condtions(cfg, algo, benchmark, task_id)
        task_id = cfg.adaptation.adaptation_task_id
        demo_num = cfg.adaptation.adapt_demo_num_each_task
        success_rate = success_rate
        return task_id, demo_num, success_rate

    def update_log_summary(log_dict, task_id, demo_num, success_rate):
        if not f'task {task_id}' in log_dict.keys():
            log_dict[f'task {task_id}'] = {}
            log_dict[f'task {task_id}'][f'demo {demo_num}'] = success_rate
        else:
            if not f'demo {demo_num}' in log_dict[f'task {task_id}']:
                log_dict[f'task {task_id}'][f'demo {demo_num}'] = success_rate
            else:
                if success_rate > log_dict[f'task {task_id}'][f'demo {demo_num}']:
                    log_dict[f'task {task_id}'][f'demo {demo_num}'] = success_rate

        return log_dict

    pre_trained_model_path = '../scripts/experiments/LIBERO_OBJECT/PreTrainMultitask/BCTransformerPolicy_seed10000/run_003/multitask_model.pth'
    adaptor_model_paths = '../scripts/experiments/LIBERO_OBJECT/PreTrainMultitask/LoraBCTPolicy_seed10000/'

    log_summary = {}

    for root, dirs, files in os.walk(adaptor_model_paths):
        for directory in dirs:
            run_path = os.path.join(root, directory)
            exp_paths = [folder for folder in glob(os.path.join(run_path, '*.pth'))]
            for exp_path in exp_paths:
                print(exp_path)
                task_id, demo_num, success_rate = evaluate_one_repo_adaptor(pre_trained_model_path, exp_path)
                log_summary = update_log_summary(log_summary, task_id, demo_num, success_rate)
                print('************************')


if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    main()
