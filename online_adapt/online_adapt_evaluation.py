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


@hydra.main(config_path="../configs", config_name="online_adaptation", version_base=None)
def main(om_cfg):
    def evaluate_one_repo_adaptor(task_id, pre_train_model_path, adaptor_model_path):
        # load the pre-trained model and adaptor model
        sd, cfg, previous_mask = torch_load_model(
            pre_train_model_path, map_location=None
        )

        lora_model_sd = torch.load(adaptor_model_path, map_location=None)

        control_seed(cfg.seed)
        cfg.folder = get_libero_path("datasets")
        cfg.bddl_folder = get_libero_path("bddl_files")
        cfg.init_states_folder = get_libero_path("init_states")

        cfg_adapt_path = os.path.join(*adaptor_model_path.split('/')[:-1], 'config.json')
        with open(cfg_adapt_path, 'r') as f:
            cfg_adapt = json.load(f)
        cfg_adapt = EasyDict(cfg_adapt)
        cfg.adaptation = cfg_adapt
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
        success_rate = evaluate_success_all_init_condtions(cfg, algo, benchmark, [task_id])

        return success_rate

    def update_log_summary(log_dict, task_id, demo_num, success_rate, exp_path):
        if not f'task {task_id}' in log_dict.keys():
            log_dict[f'task {task_id}'] = {}
            log_dict[f'task {task_id}'][f'demo {demo_num}'] = {}
            log_dict[f'task {task_id}'][f'demo {demo_num}']['success_rate'] = success_rate
            log_dict[f'task {task_id}'][f'demo {demo_num}']['path'] = exp_path

        else:
            if not f'demo {demo_num}' in log_dict[f'task {task_id}']:
                log_dict[f'task {task_id}'][f'demo {demo_num}'] = {}
                log_dict[f'task {task_id}'][f'demo {demo_num}']['success_rate'] = success_rate
                log_dict[f'task {task_id}'][f'demo {demo_num}']['path'] = exp_path
            else:
                if success_rate > log_dict[f'task {task_id}'][f'demo {demo_num}']['success_rate']:
                    log_dict[f'task {task_id}'][f'demo {demo_num}']['success_rate'] = success_rate
                    log_dict[f'task {task_id}'][f'demo {demo_num}']['path'] = exp_path

        return log_dict

    pre_trained_model_path = '../scripts/experiments/LIBERO_OBJECT/PreTrainMultitask/BCTransformerPolicy_seed10000/run_003/multitask_model.pth'

    adaptor_model_paths = os.path.join(om_cfg.exp_dir, f'demo_{om_cfg.adapt_demo_num_each_task}',
                                       f'support_{om_cfg.meta_support_num}_query_{om_cfg.meta_query_num}',
                                       f'seed_{om_cfg.seed}')
    log_summary = {}

    for root, dirs, files in os.walk(adaptor_model_paths):
        for directory in dirs:
            run_path = os.path.join(root, directory)
            exp_paths = [folder for folder in glob(os.path.join(run_path, '*.pth'))]
            finish_flag = os.path.join(run_path, 'task_9_ep_100.pth') in exp_paths and 'run_020' in run_path
            # if not finish_flag:
            #        print(run_path)
            with open(os.path.join(run_path, 'config.json'), 'r') as f:
                config = json.load(f)
            f.close()
            #     print('########################')
            if finish_flag:

                tasks_best = np.zeros(5)
                tasks_best_path = ['', '', '', '', '']
                for exp_path in exp_paths:
                    if 'ep' in exp_path:

                        task_id = int(exp_path.split('/')[-1].split('_')[1])
                        ep = exp_path.split('/')[-1].split('_')[3].split('.')[0]

                        success_rate = evaluate_one_repo_adaptor(task_id, pre_trained_model_path, exp_path)[0]
                        ind = task_id - 5
                        if success_rate > tasks_best[ind]:
                            tasks_best[ind] = success_rate
                            tasks_best_path[ind] = exp_path
                        print(f'task:{task_id}, ep:{ep}, success_rate:{success_rate}')

                print(run_path)
                print(
                    f"adpat num:{config['adapt_demo_num_each_task']}.meta_update_epochs:{config['meta_update_epochs']}.support:{config['meta_support_num']}.query:{config['meta_query_num']}")
                print(tasks_best)
                print(tasks_best_path)
                print('----------------------------------------------------')

            # for exp_path in exp_paths:
            #     print(exp_path)
            #     # task_id, demo_num, success_rate = evaluate_one_repo_adaptor(pre_trained_model_path, exp_path)
            #     # print(task_id, demo_num, success_rate)
            #     # log_summary = update_log_summary(log_summary, task_id, demo_num, success_rate, exp_path)
            #     print('************************')

    # print(log_summary)
    # with open('./log_summary.json', 'w') as f:
    #     f.write(json.dump(log_summary))
    #     f.close()


if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    main()
