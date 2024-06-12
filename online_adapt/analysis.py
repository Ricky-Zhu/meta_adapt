import glob
from warnings import filterwarnings
import sys

import torch

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
import json
from online_meta_algo import *


def mapped_features(pre_train_model_sd, lora_params_sd, cfg, task_id):
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
                load_specific_num=2
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

    all_dataset = [SequenceVLDataset(ds, emb) for (ds, emb) in
                   zip(manip_datasets, task_embs)]

    ##################################
    # model with lora definition
    cfg.policy.policy_type = 'LoraBCTPolicy'
    cfg.lifelong.algo = 'OnlineMeta'  # for creating the exp dir in base algo : sequential
    # remove the previous experiment dir so that the initialization of algo will create a new exp dir
    cfg.pop('experiment_dir')

    algo = safe_device(eval('OnlineMeta')(10, cfg, pre_train_model_sd), 'cuda')
    if lora_params_sd is not None:
        algo.policy.load_state_dict(lora_params_sd, strict=False)
    # algo.policy.previous_mask = previous_mask
    # save the configs
    algo.policy.eval()
    all_task_features = []
    for id in range(len(all_dataset)):
        task_features = []
        task_specific_data = all_dataset[id]
        task_specific_dl = DataLoader(task_specific_data, batch_size=10, shuffle=False, num_workers=4, drop_last=False)
        for i, data in enumerate(task_specific_dl):
            tensor_data = algo.map_tensor_to_device(data)
            x = algo.policy.spatial_encode(tensor_data)
            features = algo.policy.temporal_encode(x)
            features = algo.policy.policy_head(features)
            task_features.append(features)

        task_features = torch.cat(task_features, dim=0)
        shape = task_features.shape
        task_features = task_features.reshape(shape[0], -1)
        task_features = task_features.detach().cpu().numpy().tolist()
        all_task_features.append(task_features)

    with open(f'./rand.json', 'w') as f:
        json.dump(all_task_features, f)
        f.close()


if __name__ == "__main__":
    pre_trained_model_path = '../scripts/experiments/LIBERO_OBJECT/PreTrainMultitask/BCTransformerPolicy_seed10000/run_003/multitask_model.pth'

    pre_trained_model_sd, cfg, previous_mask = torch_load_model(
        pre_trained_model_path, map_location=None
    )

    lora_params_model_parent_path = '../scripts/experiments/lora_online_adaptation/demo_10/support_10_query_15/meta_update_epochs_50/random_meta_False/seed_0/'

    files = glob.glob(lora_params_model_parent_path + 'meta*.pth')

    lora_cfg_path = os.path.join(lora_params_model_parent_path, 'config.json')
    with open(lora_cfg_path, 'r') as f:
        lora_cfg = json.load(f)
        f.close()
    cfg.adaptation = lora_cfg

    # for file in files:
    #     task_id = file.split('.')[-2].split('_')[-1]
    #     lora_params_sd = torch.load(file)
    #
    #     mapped_features(pre_trained_model_sd, lora_params_sd=lora_params_sd, cfg=cfg, task_id=task_id)

    lora_path = '/home/ruiqi/projects/meta_adapt/scripts/experiments/lora_online_adaptation/demo_10/support_10_query_15/meta_update_epochs_50/random_meta_False/seed_10000/task_6_ep_30.pth'

    lora_params_sd = torch.load(lora_path)

    mapped_features(pre_trained_model_sd, lora_params_sd=None, cfg=cfg, task_id=6)
