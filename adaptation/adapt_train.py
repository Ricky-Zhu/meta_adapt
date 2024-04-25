from warnings import filterwarnings
import sys

sys.path.append("..")
filterwarnings(action='ignore', category=DeprecationWarning)
import os
import sys
from os.path import dirname, abspath

sys.path.append(dirname(dirname(abspath(__file__))))
# TODO: find a better way for this?
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import hydra
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.datasets import get_dataset, SequenceVLDataset
from libero.lifelong.utils import (
    control_seed,
    safe_device,
    torch_load_model,
)
from libero.lifelong.main import get_task_embs
from pre_training.pre_training_algo import *
from lora_parts.policy import *
import loralib as lora

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


@hydra.main(config_path="../configs", config_name="adaptation", version_base=None)
def main(adaptation_cfg):
    # define the pre-trained model path
    model_path = '../scripts/experiments/LIBERO_OBJECT/PreTrainMultitask/BCTransformerPolicy_seed10000/run_003/multitask_model.pth'

    sd, cfg, previous_mask = torch_load_model(
        model_path, map_location=None
    )

    # load specific adaptation configs
    cfg.adaptation = adaptation_cfg.adaptation

    control_seed(cfg.adaptation.seed)
    cfg.folder = get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")

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
                load_specific_num=cfg.adaptation.adapt_demo_num_each_task
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

    post_adaptation_dataset = [SequenceVLDataset(manip_datasets[cfg.adaptation.adaptation_task_id],
                                                 task_embs[cfg.adaptation.adaptation_task_id])]

    ##################################
    # model with lora definition
    cfg.policy.policy_type = 'LoraBCTPolicy'
    cfg.eval.n_eval = cfg.adaptation.n_eval  # the number of evaluations in the adaptation

    # remove the previous experiment dir so that the initialization of algo will create a new exp dir
    cfg.pop('experiment_dir')
    cfg.experiment_dir = os.path.join(cfg.adaptation.exp_dir, f'task_{cfg.adaptation.adaptation_task_id}',
                                      f'demo_{cfg.adaptation.adapt_demo_num_each_task}',
                                      f'seed_{cfg.adaptation.seed}')  # load the customized experiment dir (e.g. ./experiment/lora_adaptation/task_7)
    if not os.path.exists(cfg.experiment_dir):
        os.makedirs(cfg.experiment_dir)
    algo = safe_device(eval('PreTrainMultitask')(10, cfg), 'cuda')
    algo.policy.previous_mask = previous_mask
    algo.policy.load_state_dict(sd, strict=False)

    which_bias_train = 'lora_only' if not cfg.adaptation.train_all_bias else 'all'
    lora.mark_only_lora_as_trainable(algo.policy, bias=which_bias_train)

    # prepare experiment dir and start training
    algo.adapt(post_adaptation_dataset, benchmark, adapt_task_id=cfg.adaptation.adaptation_task_id,
               which_bias_train=which_bias_train)


if __name__ == "__main__":
    main()
