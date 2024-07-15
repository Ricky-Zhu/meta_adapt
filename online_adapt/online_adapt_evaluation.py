from warnings import filterwarnings
from omegaconf import OmegaConf

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
from libero.lifelong.main import get_task_embs
from utils.evaluation import *
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
from lora_parts.policy import *
from glob import glob
from online_meta_algo import *
from easydict import EasyDict
import hydra


class SimpleLogger():
    def __init__(self, logger_path, start_log):
        self.logger_path = logger_path
        self.start_log = start_log

    def write_and_print(self, sentence, to_print=False):
        if self.start_log:
            with open(self.logger_path, "a") as f:
                f.write(sentence + '\n')
        if to_print:
            print(sentence)


@hydra.main(config_path="../configs", config_name="online_adaptation", version_base=None)
def main(om_cfg):
    def evaluate_one_repo_adaptor(task_id, pre_train_model_path, adaptor_model_path, cfg_adapt):
        # load the pre-trained model and adaptor model
        sd, cfg, previous_mask = torch_load_model(
            pre_train_model_path, map_location=None
        )

        lora_model_sd = torch.load(adaptor_model_path, map_location=None)

        control_seed(om_cfg.seed)
        cfg.folder = get_libero_path("datasets")
        cfg.bddl_folder = get_libero_path("bddl_files")
        cfg.init_states_folder = get_libero_path("init_states")

        cfg_adapt = EasyDict(cfg_adapt)
        cfg.adaptation = cfg_adapt
        cfg.policy.policy_type = 'LoraBCTPolicy'
        algo = safe_device(eval('OnlineMeta')(10, cfg, sd), 'cuda')
        algo.policy.previous_mask = previous_mask

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

        # cfg.eval.n_eval=20 if want to set a different eval num as that in the exp
        cfg.eval.n_eval = 50
        cfg.eval.use_mp = True
        success_rate = evaluate_success_all_init_condtions(cfg, algo, benchmark, [task_id])

        return success_rate

    logger_path = f'../online_adapt/logger_{om_cfg.seed}.txt'
    logger = SimpleLogger(logger_path=logger_path, start_log=True)
    logger.write_and_print('start evaluation', to_print=True)
    pre_trained_model_path = om_cfg.pre_trained_model_path
    benchmark_name = om_cfg.pre_trained_model_path.split('/')[-5]
    adaptor_model_paths = os.path.join(om_cfg.exp_dir, benchmark_name,
                                       f'demo_{om_cfg.adapt_demo_num_each_task}',
                                       f'meta_epoch_{om_cfg.meta_update_epochs}',
                                       f'seed_{om_cfg.seed}')

    files = os.listdir(adaptor_model_paths)
    finish_flag = f'task_9_ep_{om_cfg.n_epochs}.pth' in files
    if finish_flag:
        with open(os.path.join(adaptor_model_paths, 'config.json'), 'r') as f:
            config = json.load(f)
            f.close()
        tasks_best = np.zeros(5)
        tasks_best_path = ['', '', '', '', '']
        for exp_path in files:
            if 'ep' in exp_path and '_0' not in exp_path:

                task_id = int(exp_path.split('_')[1])
                ep = int(exp_path.split('_')[-1].split('.')[0])

                abs_exp_path = os.path.join(adaptor_model_paths, exp_path)
                success_rate = evaluate_one_repo_adaptor(task_id, pre_trained_model_path, abs_exp_path, config)[0]
                ind = task_id - 5
                if success_rate > tasks_best[ind]:
                    tasks_best[ind] = success_rate
                    tasks_best_path[ind] = exp_path
                print(f'task:{task_id}, ep:{ep}, success_rate:{success_rate}')

        config_info = f"suite:{benchmark_name}, policy:{om_cfg.policy_type},seed:{om_cfg.seed}, demo num:{om_cfg.adapt_demo_num_each_task}\n"
        logger.write_and_print(config_info, to_print=True)
        logger.write_and_print(tasks_best, to_print=True)
        logger.write_and_print(tasks_best_path, to_print=True)
        logger.write_and_print('----------------------------------------------------', to_print=True)

        # save the log
        log_save_path = os.path.join(adaptor_model_paths, 'performance.txt')
        with open(log_save_path, 'w') as f:
            f.write(config_info)
            f.write(str(tasks_best))
            f.write('\n')
            f.write(str(tasks_best_path))


if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    main()
