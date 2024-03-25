import os
from pre_training_algo import PreTrainMultitask
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import multiprocessing
import pprint
import hydra
import wandb
import yaml
from easydict import EasyDict
from omegaconf import OmegaConf
from libero.libero import get_libero_path
from libero.lifelong.algos import get_algo_class, get_algo_list
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')
from libero.lifelong.utils import (
    NpEncoder,
    compute_flops,
    control_seed,
    safe_device,
    torch_load_model,
    create_experiment_dir,
    get_task_embs,
)
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
from utils.task_creation import create_tasks


@hydra.main(config_path="../configs", config_name="pre_training", version_base=None)
def main(hydra_cfg):
    # preprocessing
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    # print configs to terminal
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(cfg)

    # control seed
    control_seed(cfg.seed)

    # prepare lifelong learning
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")

    pre_training_dataset, post_adaptation_dataset, benchmark_instance, shape_meta = create_tasks(cfg)

    print("\n=================== Pretraining ===================")
    print(f" Name: {benchmark_instance.name}")
    print(f" # Tasks: {cfg.task_creation.pre_training_num}")

    for i in range(cfg.task_creation.pre_training_num):
        print(f"    - Task {i + 1}:")
        print(f"        {benchmark_instance.get_task(i).language}")
    print("=======================================================================\n")

    # prepare experiment and update the config
    create_experiment_dir(cfg)
    cfg.shape_meta = shape_meta

    if cfg.use_wandb:
        wandb.init(project="libero_pretraining", config=cfg)
        wandb.run.name = cfg.experiment_name

    # result_summary = {
    #     "L_conf_mat": np.zeros((n_manip_tasks, n_manip_tasks)),  # loss confusion matrix
    #     "S_conf_mat": np.zeros((n_manip_tasks, n_manip_tasks)),  # success confusion matrix
    #     "L_fwd": np.zeros((n_manip_tasks,)),  # loss AUC, how fast the agent learns
    #     "S_fwd": np.zeros((n_manip_tasks,)),  # success AUC, how fast the agent succeeds
    # }

    # if cfg.eval.save_sim_states:
    #     # for saving the evaluate simulation states, so we can replay them later
    #     for k in range(n_manip_tasks):
    #         for p in range(k + 1):  # for testing task p when the agent learns to task k
    #             result_summary[f"k{k}_p{p}"] = [[] for _ in range(cfg.eval.n_eval)]
    #         for e in range(
    #                 cfg.train.n_epochs + 1
    #         ):  # for testing task k at the e-th epoch when the agent learns on task k
    #             if e % cfg.eval.eval_every == 0:
    #                 result_summary[f"k{k}_e{e // cfg.eval.eval_every}"] = [
    #                     [] for _ in range(cfg.eval.n_eval)
    #                 ]

    # define lifelong algorithm, n_task here does not matter
    algo = safe_device(get_algo_class(cfg.lifelong.algo)(n_tasks=1, cfg=cfg), cfg.device)
    # if cfg.pretrain_model_path != "":  # load a pretrained model if there is any
    #     try:
    #         algo.policy.load_state_dict(torch_load_model(cfg.pretrain_model_path)[0])
    #     except:
    #         print(
    #             f"[error] cannot load pretrained model from {cfg.pretrain_model_path}"
    #         )
    #         sys.exit(0)

    print(f"[info] start lifelong learning with algo {cfg.lifelong.algo}")
    # GFLOPs, MParams = compute_flops(algo, pre_training_dataset[0], cfg)
    # print(f"[info] policy has {GFLOPs:.1f} GFLOPs and {MParams:.1f} MParams\n")

    # save the experiment config file, so we can resume or replay later
    with open(os.path.join(cfg.experiment_dir, "config.json"), "w") as f:
        json.dump(cfg, f, cls=NpEncoder, indent=4)

    algo.train()
    s_fwd, l_fwd = algo.learn_all_tasks(pre_training_dataset, benchmark_instance)

    # evalute on all seen tasks at the end if eval.eval is true
    # if cfg.eval.eval:
    #     L = evaluate_loss(cfg, algo, benchmark, datasets)
    #     S = evaluate_success(
    #         cfg=cfg,
    #         algo=algo,
    #         benchmark=benchmark,
    #         task_ids=list(range(n_manip_tasks)),
    #         result_summary=result_summary if cfg.eval.save_sim_states else None,
    #     )
    #
    #     result_summary["L_conf_mat"][-1] = L
    #     result_summary["S_conf_mat"][-1] = S
    #
    #     if cfg.use_wandb:
    #         wandb.run.summary["success_confusion_matrix"] = result_summary[
    #             "S_conf_mat"
    #         ]
    #         wandb.run.summary["loss_confusion_matrix"] = result_summary[
    #             "L_conf_mat"
    #         ]
    #         wandb.run.summary["fwd_transfer_success"] = result_summary["S_fwd"]
    #         wandb.run.summary["fwd_transfer_loss"] = result_summary["L_fwd"]
    #         wandb.run.summary.update()
    #
    #     print(("[All task loss ] " + " %4.2f |" * n_tasks) % tuple(L))
    #     print(("[All task succ.] " + " %4.2f |" * n_tasks) % tuple(S))
    #
    #     torch.save(result_summary, os.path.join(cfg.experiment_dir, f"result.pt"))

    print("[info] finished learning\n")
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn'
    # if multiprocessing.get_start_method(allow_none=True) != "spawn":
    #     multiprocessing.set_start_method("spawn", force=True)
    main()
