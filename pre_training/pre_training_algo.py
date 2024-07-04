import os
from warnings import filterwarnings

filterwarnings(action='ignore', category=DeprecationWarning)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler

from libero.lifelong.algos.base import Sequential, register_algo
from libero.lifelong.metric import *
from libero.lifelong.models import *
from libero.lifelong.utils import *
import loralib as lora
import multiprocessing
from torchvision import transforms


class PreTrainMultitask(Sequential):
    """
    The multitask learning baseline/upperbound.
    """

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        if self.cfg.multi_gpu:
            self.policy = nn.DataParallel(self.policy)

    def learn_all_tasks(self, datasets, benchmark):
        self.start_task(-1)
        concat_dataset = ConcatDataset(datasets)

        # learn on all tasks, only used in multitask learning
        model_checkpoint_name = os.path.join(
            self.experiment_dir, f"multitask_model.pth"
        )
        pretrain_tasks = self.cfg.task_creation.select_tasks

        train_dataloader = DataLoader(
            concat_dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            sampler=RandomSampler(concat_dataset),
            persistent_workers=True,
        )

        prev_success_rate = -1.0
        cumulated_counter = 0.0
        idx_at_best_succ = 0
        successes = []
        losses = []

        # start training
        for epoch in range(1, self.cfg.train.n_epochs + 1):

            t0 = time.time()
            # if epoch > 0 or (self.cfg.pretrain):  # update
            self.policy.train()
            training_loss = 0.0
            for (idx, data) in enumerate(train_dataloader):
                loss = self.observe(data)
                training_loss += loss
            training_loss /= len(train_dataloader)

            t1 = time.time()

            print(
                f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1 - t0) / 60:4.2f}"
            )

            if epoch % self.cfg.eval.eval_every == 0:
                t0 = time.time()
                self.policy.eval()

                if self.cfg.eval.save_ep_model:
                    model_checkpoint_name_ep = os.path.join(
                        self.experiment_dir, f"multitask_model_ep{epoch}.pth"
                    )
                    if self.cfg.multi_gpu:
                        torch_save_model(self.policy.module, model_checkpoint_name_ep, cfg=self.cfg)
                    else:
                        torch_save_model(self.policy, model_checkpoint_name_ep, cfg=self.cfg)
                losses.append(training_loss)

                # for multitask learning, we provide an option whether to evaluate
                # the agent once every eval_every epochs on all tasks, note that
                # this can be quite computationally expensive. Nevertheless, we
                # save the checkpoints, so users can always evaluate afterwards.
                if self.cfg.eval.eval:
                    success_rates = evaluate_pretrain_multitask_training_success(
                        self.cfg, self, benchmark, pretrain_tasks
                    )
                    success_rate = np.mean(success_rates)

                    successes.append(success_rate)

                    if prev_success_rate < success_rate and (not self.cfg.pretrain):
                        torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg)
                        prev_success_rate = success_rate
                        idx_at_best_succ = len(losses) - 1

                    t1 = time.time()

                    cumulated_counter += 1.0
                    ci = confidence_interval(success_rate, self.cfg.eval.n_eval)
                    tmp_successes = np.array(successes)
                    tmp_successes[idx_at_best_succ:] = successes[idx_at_best_succ]

                    if self.cfg.lifelong.eval_in_train:
                        print(
                            f"[info] Epoch: {epoch:3d} | succ: {success_rate:4.2f} ± {ci:4.2f} | best succ: {prev_success_rate} "
                            + f"| succ. AoC {tmp_successes.sum() / cumulated_counter:4.2f} | time: {(t1 - t0) / 60:4.2f}",
                            flush=True,
                        )

        self.end_task(concat_dataset, -1, benchmark)

        # return the metrics regarding forward transfer
        losses = np.array(losses)
        successes = np.array(successes)
        auc_checkpoint_name = os.path.join(
            self.experiment_dir, f"multitask_auc.log"
        )
        torch.save(
            {
                "success": successes,
                "loss": losses,
            },
            auc_checkpoint_name,
        )

        if self.cfg.lifelong.eval_in_train:
            loss_at_best_succ = losses[idx_at_best_succ]
            success_at_best_succ = successes[idx_at_best_succ]
            losses[idx_at_best_succ:] = loss_at_best_succ
            successes[idx_at_best_succ:] = success_at_best_succ

    def adapt(self, adapt_datasets, benchmark, adapt_task_id, which_bias_train):
        self.start_task(-1)
        concat_dataset = ConcatDataset(adapt_datasets)

        # learn on all tasks, only used in multitask learning
        model_checkpoint_name = os.path.join(
            self.experiment_dir, f"lora_model.pth"
        )
        adapt_task = [adapt_task_id]

        train_dataloader = DataLoader(
            concat_dataset,
            batch_size=self.cfg.adaptation.batch_size,
            num_workers=self.cfg.train.num_workers,
            sampler=RandomSampler(concat_dataset),
            persistent_workers=True,
        )

        prev_success_rate = -1.0
        # best_state_dict = self.policy.state_dict()  # currently save the best model

        # for evaluate how fast the agent learns on current task, this corresponds
        # to the area under success rate curve on the new task.
        cumulated_counter = 0.0
        idx_at_best_succ = 0
        successes = []
        losses = []

        # start training
        for epoch in range(1, self.cfg.adaptation.n_epochs + 1):

            t0 = time.time()

            self.policy.train()
            training_loss = 0.0
            for (idx, data) in enumerate(train_dataloader):
                loss = self.observe(data)
                training_loss += loss
            training_loss /= len(train_dataloader)

            t1 = time.time()

            print(
                f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1 - t0) / 60:4.2f}"
            )

            if epoch % self.cfg.adaptation.eval_every == 0:
                t0 = time.time()
                model_checkpoint_name_ep = os.path.join(
                    self.experiment_dir, f"lora_model_ep{epoch}.pth"
                )

                # only save the lora parameters
                torch.save({
                    "state_dict": lora.lora_state_dict(self.policy, bias=which_bias_train),
                    "cfg": self.cfg,
                }, model_checkpoint_name_ep)

                losses.append(training_loss)

                if self.cfg.adaptation.eval:
                    self.policy.eval()

                    success_rates = self.evaluate_during_adapt(self.cfg, self, benchmark, adapt_task)
                    success_rate = np.mean(success_rates)

                    successes.append(success_rate)

                    if prev_success_rate < success_rate and (not self.cfg.pretrain):
                        # torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg)
                        torch.save({
                            "state_dict": lora.lora_state_dict(self.policy, bias=which_bias_train),
                            "cfg": self.cfg,
                        }, model_checkpoint_name)
                        prev_success_rate = success_rate
                        idx_at_best_succ = len(losses) - 1

                    t1 = time.time()

                    cumulated_counter += 1.0
                    ci = confidence_interval(success_rate, self.cfg.eval.n_eval)
                    tmp_successes = np.array(successes)
                    tmp_successes[idx_at_best_succ:] = successes[idx_at_best_succ]

                    if self.cfg.adaptation.eval_in_train:
                        print(
                            f"[info] Epoch: {epoch:3d} | succ: {success_rate:4.2f} ± {ci:4.2f} | best succ: {prev_success_rate} "
                            + f"| succ. AoC {tmp_successes.sum() / cumulated_counter:4.2f} | time: {(t1 - t0) / 60:4.2f}",
                            flush=True,
                        )

    def start_task(self, task):
        """
        What the algorithm does at the beginning of learning each lifelong task.
        """

        try:
            self.current_task = task
            # initialize the optimizer and scheduler
            self.optimizer = eval(self.cfg.adaptation.optim_name)(
                self.policy.parameters(), **self.cfg.adaptation.optim_kwargs
            )
        except:
            super().start_task(task)

    def evaluate_during_adapt(self, cfg, algo, benchmark, adapt_task_id):

        suc_rates = evaluate_pretrain_multitask_training_success(
            cfg, algo, benchmark, adapt_task_id
        )
        return suc_rates
