from libero.libero import benchmark, get_libero_path, set_libero_default_path
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from termcolor import colored
from libero.lifelong.datasets import get_dataset, SequenceVLDataset, GroupedTaskDataset
from libero.lifelong.utils import (
    NpEncoder,
    compute_flops,
    control_seed,
    safe_device,
    torch_load_model,
    create_experiment_dir,
    get_task_embs,
)


def create_tasks(cfg):
    benchmark_root_path = get_libero_path("benchmark_root")
    init_states_default_path = get_libero_path("init_states")
    datasets_default_path = get_libero_path("datasets")
    bddl_files_default_path = get_libero_path("bddl_files")

    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[cfg.task_creation.task_suite](
        cfg.task_creation.task_order)
    num_tasks = benchmark_instance.get_num_tasks()

    manip_datasets = []
    descriptions = []
    shape_meta = None

    # first handle the pre-training dataset
    for i in range(cfg.task_creation.pre_training_num):
        # currently we assume tasks from same benchmark have the same shape_meta
        try:
            task_i_dataset, shape_meta = get_dataset(
                dataset_path=os.path.join(
                    datasets_default_path, benchmark_instance.get_task_demonstration(i)
                ),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=(i == 0),
                seq_len=cfg.data.seq_len,
                # load_specific_num=cfg.task_creation.adapt_demo_num_each_task
            )
        except Exception as e:
            print(
                f"[error] failed to load task {i} name {benchmark_instance.get_task_names()[i]}"
            )
            print(f"[error] {e}")
        # add language to the vision dataset, hence we call vl_dataset
        task_description = benchmark_instance.get_task(i).language
        descriptions.append(task_description)
        manip_datasets.append(task_i_dataset)

    # now handle the dataset of adaptation
    for i in range(cfg.task_creation.pre_training_num, num_tasks):
        # currently we assume tasks from same benchmark have the same shape_meta
        try:
            task_i_dataset, shape_meta = get_dataset(
                dataset_path=os.path.join(
                    datasets_default_path, benchmark_instance.get_task_demonstration(i)
                ),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=(i == 0),
                seq_len=cfg.data.seq_len,
                load_specific_num=cfg.task_creation.adapt_demo_num_each_task
            )
        except Exception as e:
            print(
                f"[error] failed to load task {i} name {benchmark_instance.get_task_names()[i]}"
            )
            print(f"[error] {e}")
        # add language to the vision dataset, hence we call vl_dataset
        task_description = benchmark_instance.get_task(i).language
        descriptions.append(task_description)
        manip_datasets.append(task_i_dataset)

    # the description embedding model can be changed in cfg.task_embedding_format
    task_embs = get_task_embs(cfg, descriptions)
    benchmark_instance.set_task_embs(task_embs)

    pre_training_dataset = [SequenceVLDataset(ds, emb) for (ds, emb) in
                            zip(manip_datasets[:cfg.task_creation.pre_training_num],
                                task_embs[:cfg.task_creation.pre_training_num])]
    post_adaptation_dataset = [SequenceVLDataset(ds, emb) for (ds, emb) in
                               zip(manip_datasets[cfg.task_creation.pre_training_num:],
                                   task_embs[cfg.task_creation.pre_training_num:])]

    return pre_training_dataset, post_adaptation_dataset, benchmark_instance, shape_meta


if __name__ == "__main__":
    import hydra
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    import yaml
    from easydict import EasyDict

    initialize(config_path="../configs")
    hydra_cfg = compose(config_name="config")
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    pre_training_dataset, post_adaptation_dataset, benchmark_instance, shape_meta = create_tasks(cfg)
    print('finish loading')
