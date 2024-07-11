from warnings import filterwarnings

filterwarnings(action='ignore', category=DeprecationWarning)
import os
import sys
from os.path import dirname, abspath
import multiprocessing

sys.path.append(dirname(dirname(abspath(__file__))))
# TODO: find a better way for this?
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import robomimic.utils.file_utils as FileUtils
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
import hydra
from utils.evaluation import *
import robomimic.utils.obs_utils as ObsUtils
from lora_parts.policy import *
from glob import glob
from pre_training.pre_training_algo import *
from termcolor import colored


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


@hydra.main(config_path="../configs", config_name="adaptation", version_base=None)
def main(adapt_cfg):
    def evaluate_one_repo_adaptor(pre_train_model_path, adaptor_model_path, eval_ind):
        # load the pre-trained model and adaptor model
        sd, pre_train_cfg, previous_mask = torch_load_model(
            pre_train_model_path, map_location=None
        )

        # get the cfg
        model_dict = torch.load(adaptor_model_path, map_location=None)
        cfg = model_dict['cfg']
        lora_model_sd = model_dict['state_dict']

        adapt_cfg = cfg.adaptation

        control_seed(adapt_cfg.seed)
        cfg.folder = get_libero_path("datasets")
        cfg.bddl_folder = get_libero_path("bddl_files")
        cfg.init_states_folder = get_libero_path("init_states")

        algo = safe_device(eval('PreTrainMultitask')(10, cfg), 'cuda')
        algo.policy.previous_mask = previous_mask

        algo.policy.load_state_dict(sd, strict=False)
        algo.policy.load_state_dict(lora_model_sd, strict=False)

        ##########################

        # get the benchmark the task belongs to
        benchmark = get_benchmark(adapt_cfg.adaption_suite)()
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
        if adapt_cfg.adapt_all:
            task_ids = list(range(benchmark.n_tasks))
        else:
            task_ids = adapt_cfg.adaptation_tasks

        if eval_ind == 0:
            print(f'adapt on {len(task_ids)} tasks from {cfg.adaptation.adaption_suite}')
            for task in task_ids:
                print(f'task {task}: {descriptions[task]}')

        cfg.eval.n_eval = 20
        success_rate = evaluate_success_all_init_condtions(cfg, algo, benchmark, task_ids)
        demo_num = cfg.adaptation.adapt_demo_num_each_task
        return task_ids, demo_num, success_rate

    base_path = os.path.join(adapt_cfg.exp_dir, adapt_cfg.adaption_suite, adapt_cfg.policy_type,
                             f'demo_{adapt_cfg.adapt_demo_num_each_task}',
                             f'seed_{adapt_cfg.seed}')

    logger = SimpleLogger(logger_path=adapt_cfg.logger_path, start_log=adapt_cfg.log)
    logger.write_and_print('start evaluation', to_print=True)

    newest_run = glob(os.path.join(base_path, 'run_*'))
    newest_run.sort()
    newest_run = newest_run[-1]

    pth_paths = glob(os.path.join(newest_run, "*pth"))
    best_suc = -1.0
    best_ep = None
    for i in range(len(pth_paths)):
        path = pth_paths[i]
        task_id, demo_num, success_rate = evaluate_one_repo_adaptor(adapt_cfg.pre_trained_model_path, path, eval_ind=i)
        print(task_id, demo_num, success_rate, path)
        success_rate = np.mean(success_rate)
        print(f'avg:{success_rate}')
        if success_rate > best_suc:
            best_suc = success_rate
            best_ep = path
    logger.write_and_print(f'evaluate on tasks {task_id}', to_print=True)
    final_sentence = f'best suc:{best_suc}, best_ep:{best_ep}'
    logger.write_and_print(final_sentence)
    logger.write_and_print('----------------------------')
    print(colored(final_sentence, 'red'))
    with open(os.path.join(newest_run, 'performance.txt'), 'w') as f:
        f.write(final_sentence)
    f.close()


if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    main()
