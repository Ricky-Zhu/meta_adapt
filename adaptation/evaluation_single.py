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


@hydra.main(config_path="../configs", config_name="adaptation", version_base=None)
def main(adapt_cfg):
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

    pre_trained_model_path = '../scripts/experiments/LIBERO_OBJECT/PreTrainMultitask/BCTransformerPolicy_seed10000/run_003/multitask_model.pth'
    adaptor_model_paths = os.path.join(adapt_cfg.adaptation.exp_dir, f'task_{adapt_cfg.adaptation.adaptation_task_id}',
                                       f'demo_{adapt_cfg.adaptation.adapt_demo_num_each_task}')

    for root, dirs, files in os.walk(adaptor_model_paths):

        best_suc = -1.0
        best_ep = None
        for file in files:
            if 'ep' in file and 'pth' in file:
                adaptor_path = os.path.join(root, file)
                task_id, demo_num, success_rate = evaluate_one_repo_adaptor(pre_trained_model_path, adaptor_path)
                print(task_id, demo_num, success_rate, file)
                if success_rate > best_suc:
                    best_suc = success_rate
                    best_ep = adaptor_path
        print(f'task:{task_id}, best suc:{best_suc}, best_ep:{best_ep}')


if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    main()
