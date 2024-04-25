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

from online_meta_algo import *


@hydra.main(config_path="../configs", config_name="online_adaptation", version_base=None)
def main(online_adaptation_cfg):
    # define the pre-trained model path
    model_path = '../scripts/experiments/LIBERO_OBJECT/PreTrainMultitask/BCTransformerPolicy_seed10000/run_003/multitask_model.pth'

    sd, cfg, previous_mask = torch_load_model(
        model_path, map_location=None
    )

    # load specific online adaptation configs
    cfg.adaptation = online_adaptation_cfg

    control_seed(cfg.seed)
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

    print(f'Use {cfg.adaptation.adapt_demo_num_each_task} demos')
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

    pre_train_dataset = [SequenceVLDataset(ds, emb) for (ds, emb) in
                         zip(manip_datasets[:cfg.adaptation.post_adaptation_start_id],
                             task_embs[:cfg.adaptation.post_adaptation_start_id])]

    post_adaptation_dataset = [SequenceVLDataset(ds, emb) for (ds, emb) in
                               zip(manip_datasets[cfg.adaptation.post_adaptation_start_id:],
                                   task_embs[cfg.adaptation.post_adaptation_start_id:])]
    ##################################
    # model with lora definition
    cfg.policy.policy_type = 'LoraBCTPolicy'
    cfg.lifelong.algo = 'OnlineMeta'  # for creating the exp dir in base algo : sequential
    # remove the previous experiment dir so that the initialization of algo will create a new exp dir
    cfg.pop('experiment_dir')

    algo = safe_device(eval('OnlineMeta')(10, cfg, sd), 'cuda')
    # algo.policy.previous_mask = previous_mask
    # save the configs
    config_path = os.path.join(cfg.experiment_dir, 'config.json')
    yaml_config = OmegaConf.to_yaml(online_adaptation_cfg)
    adapt_cfg = EasyDict(yaml.safe_load(yaml_config))
    with open(config_path, 'w') as f:
        json.dump(adapt_cfg, f, cls=NpEncoder, indent=4)
    # online adapting
    algo.online_adapt(benchmark, pre_train_dataset, post_adaptation_dataset)


if __name__ == "__main__":
    main()
