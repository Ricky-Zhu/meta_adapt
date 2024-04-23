import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
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
from policy import *
from torch import autograd
from utils.online_adapt_utils import clone_module, update_module
import json
from libero.lifelong.utils import NpEncoder
from omegaconf import OmegaConf
import yaml
from easydict import EasyDict


class OnlineMeta(Sequential):
    """
    algo: Online meta learn a set of initial parameters for LORA module
    """

    def __init__(self, n_tasks, cfg, pre_trained_model_sd, meta_params_path=None):
        super().__init__(n_tasks=n_tasks, cfg=cfg)
        # initialize the meta lora parameters and load the pre-trained model parameters
        self.policy.load_state_dict(pre_trained_model_sd, strict=False)

        # load the meta_params
        if meta_params_path is not None:
            raise NotImplementedError
        lora.mark_only_lora_as_trainable(self.policy, bias=self.cfg.adaptation.bias_training_type)

        self.diff_params_names = [name for name, p in self.policy.named_parameters() if p.requires_grad]

        self.meta_update_inner_lr = {
            k: torch.tensor(cfg.adaptation.meta_update_inner_lr,
                            requires_grad=cfg.adaptation.learn_meta_update_inner_lrs,
                            device=cfg.device)
            for k in self.diff_params_names
        }

        self.meta_optimizer = eval(self.cfg.train.optimizer.name)(
            list(self.policy.parameters()) + list(self.meta_update_inner_lr.values()), **self.cfg.train.optimizer.kwargs
        )


    def online_adapt(self, benchmark, pre_train_dataset, post_adaptation_dataset):
        for task in range(len(post_adaptation_dataset)):
            self.start_task(task + self.cfg.adaptation.post_adaptation_start_id)
            existing_dataset = pre_train_dataset + post_adaptation_dataset[:task + 1]
            self.meta_update(existing_dataset=existing_dataset)
            self.update_procedure(task_specific_dataset=post_adaptation_dataset[task])
            self.load_meta_lora_params()  # load the lora params which are before the fine tunning

    def meta_update(self, existing_dataset):
        existing_task_num = len(existing_dataset)
        batch_size = self.cfg.adaptation.meta_support_num + self.cfg.adaptation.meta_query_num
        iter_dl_list = self.turn_to_iters(existing_dataset, batch_size)

        print('--------- Meta Learning Phase ---------------')
        for ep in range(self.cfg.adaptation.meta_update_epochs):
            ep_query_loss = []
            if self.cfg.adaptation.random_meta:
                for i in range(len(iter_dl_list)):  # for being fair with comparing iterating over all datasets
                    # select task each time randomly
                    task_id = random.randint(0, existing_task_num - 1)
                    try:
                        data = next(iter_dl_list[task_id])
                    except:
                        iter_dl_list[i] = iter(DataLoader(existing_dataset[i], batch_size=batch_size, shuffle=True,
                                                          num_workers=self.cfg.adaptation.num_workers))
                        data = next(iter_dl_list[i])
                    support_data, query_data = self.split_support_query(data)
                    adapted_policy_net = self._meta_inner_step(support_data)
                    loss = self.meta_val(adapted_policy_net, query_data)
                    ep_query_loss.append(loss)

            else:
                # iterate over all task dataset
                for i in range(len(iter_dl_list)):
                    try:
                        data = next(iter_dl_list[i])
                    except:
                        iter_dl_list[i] = iter(DataLoader(existing_dataset[i], batch_size=batch_size, shuffle=True,
                                                          num_workers=self.cfg.adaptation.num_workers))
                        data = next(iter_dl_list[i])
                    support_data, query_data = self.split_support_query(data)
                    adapted_policy_net = self._meta_inner_step(support_data)
                    loss = self.meta_val(adapted_policy_net, query_data)
                    ep_query_loss.append(loss)

            # TODO logging the loss
            avg_loss = sum(ep_query_loss) / len(ep_query_loss)
            if ep % self.cfg.adaptation.meta_display_interval == 0:
                print(f'epoch: {ep}, meta query loss: {avg_loss}, current task: {self.current_task}')

        print('##################################')
        # TODO save the lora parameters of the policy net as meta lora parameters
        self.save_meta_lora_params()

    def update_procedure(self, task_specific_dataset):
        train_dataloader = DataLoader(
            task_specific_dataset, batch_size=self.cfg.adaptation.update_procedure_batch_size, shuffle=True,
            num_workers=self.cfg.adaptation.num_workers)

        print('--------- Update Procedure Phase ---------------')
        for epoch in range(0, self.cfg.adaptation.n_epochs + 1):

            t0 = time.time()
            if epoch > 0:
                self.policy.train()
                training_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.observe(data)
                    training_loss += loss
                training_loss /= len(train_dataloader)
            else:  # just evaluate the zero-shot performance on 0-th epoch
                training_loss = 0.0
                #
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.eval_observe(data)
                    training_loss += loss
                training_loss /= len(train_dataloader)
            t1 = time.time()

            print(
                f"[info] Task: {self.current_task}| Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1 - t0) / 60:4.2f}"
            )
            if epoch % self.cfg.adaptation.save_interval == 0:
                print(
                    f"[info] Task: {self.current_task}| Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1 - t0) / 60:4.2f}"
                )
                self.save_task_specific_lora_params(ep=epoch)
        print('##############################')

    def save_task_specific_lora_params(self, ep):
        path = os.path.join(self.cfg.experiment_dir, f'task_{self.current_task}_ep_{ep}.pth')
        torch.save(lora.lora_state_dict(self.policy, bias=self.cfg.adaptation.bias_training_type), path)

    def save_meta_lora_params(self):
        self.meta_lora_params = lora.lora_state_dict(self.policy, bias=self.cfg.adaptation.bias_training_type)
        path = os.path.join(self.cfg.experiment_dir, f'meta_params_task_{self.current_task}.pth')
        torch.save(self.meta_lora_params, path)

    def load_meta_lora_params(self):
        self.policy.load_state_dict(self.meta_lora_params, strict=False)

    def turn_to_iters(self, datasets, batch_size):
        iters_list = []

        for i in range(len(datasets)):
            temp_dl = DataLoader(datasets[i], batch_size=batch_size, shuffle=True,
                                 num_workers=self.cfg.adaptation.num_workers)
            iters_list.append(iter(temp_dl))
        return iters_list

    def split_support_query(self, data):
        support_data = {
            'actions': data['actions'][:self.cfg.adaptation.meta_support_num],
            'task_emb': data['task_emb'][:self.cfg.adaptation.meta_support_num],
            'obs': {'agentview_rgb': data['obs']['agentview_rgb'][:self.cfg.adaptation.meta_support_num],
                    'eye_in_hand_rgb': data['obs']['eye_in_hand_rgb'][:self.cfg.adaptation.meta_support_num],
                    'gripper_states': data['obs']['gripper_states'][:self.cfg.adaptation.meta_support_num],
                    'joint_states': data['obs']['joint_states'][:self.cfg.adaptation.meta_support_num]
                    }
        }
        query_data = {
            'actions': data['actions'][self.cfg.adaptation.meta_support_num:],
            'task_emb': data['task_emb'][self.cfg.adaptation.meta_support_num:],
            'obs': {'agentview_rgb': data['obs']['agentview_rgb'][self.cfg.adaptation.meta_support_num:],
                    'eye_in_hand_rgb': data['obs']['eye_in_hand_rgb'][self.cfg.adaptation.meta_support_num:],
                    'gripper_states': data['obs']['gripper_states'][self.cfg.adaptation.meta_support_num:],
                    'joint_states': data['obs']['joint_states'][self.cfg.adaptation.meta_support_num:]
                    }
        }
        return support_data, query_data

    def meta_val(self, adapted_policy_net, query_data):
        data = self.map_tensor_to_device(query_data)
        self.meta_optimizer.zero_grad()
        loss = self.loss_scale * adapted_policy_net.compute_loss(data)
        loss.backward()

        self.meta_optimizer.step()

        return loss.item()

    def _meta_inner_step(self, data):
        target_policy_net = clone_module(self.policy)
        data = self.map_tensor_to_device(data)

        loss = target_policy_net.compute_loss(data)
        diff_params = [p for name, p in target_policy_net.named_parameters() if name in self.diff_params_names]
        grads = torch.autograd.grad(loss, diff_params, create_graph=True)
        gradients = []
        grad_counter = 0

        # Handles gradients for non-differentiable parameters
        for name, param in target_policy_net.named_parameters():
            if name in self.diff_params_names:
                gradient = grads[grad_counter]
                grad_counter += 1
            else:
                gradient = None
            gradients.append(gradient)

        if gradients is not None:
            names = []
            params = []
            for name, param in target_policy_net.named_parameters():
                names.append(name)
                params.append(param)
            if not len(gradients) == len(list(params)):
                msg = 'WARNING:maml_update(): Parameters and gradients have different length. ('
                msg += str(len(params)) + ' vs ' + str(len(grads)) + ')'
                print(msg)
            for p, n, g in zip(params, names, gradients):
                if g is not None:
                    # p.update = - 0.4 * g
                    p.update = - self.meta_update_inner_lr[n] * g
        target_policy_net = update_module(target_policy_net)

        return target_policy_net

    def start_task(self, task):
        """
        What the algorithm does at the beginning of learning each lifelong task.
        """
        self.current_task = task

        # initialize the optimizer and scheduler
        self.optimizer = eval(self.cfg.train.optimizer.name)(
            self.policy.parameters(), **self.cfg.train.optimizer.kwargs
        )

        self.scheduler = None
        if self.cfg.train.scheduler is not None:
            self.scheduler = eval(self.cfg.train.scheduler.name)(
                self.optimizer,
                T_max=self.cfg.adaptation.n_epochs,
                **self.cfg.train.scheduler.kwargs,
            )

    # def adapt(self, adapt_datasets, benchmark, adapt_task_id, which_bias_train):
    #     self.start_task(-1)
    #     concat_dataset = ConcatDataset(adapt_datasets)
    #
    #     # learn on all tasks, only used in multitask learning
    #     model_checkpoint_name = os.path.join(
    #         self.experiment_dir, f"lora_model.pth"
    #     )
    #     adapt_task = [adapt_task_id]
    #
    #     train_dataloader = DataLoader(
    #         concat_dataset,
    #         batch_size=self.cfg.train.batch_size,
    #         num_workers=self.cfg.train.num_workers,
    #         sampler=RandomSampler(concat_dataset),
    #         persistent_workers=True,
    #     )
    #
    #     prev_success_rate = -1.0
    #     best_state_dict = self.policy.state_dict()  # currently save the best model
    #
    #     # for evaluate how fast the agent learns on current task, this corresponds
    #     # to the area under success rate curve on the new task.
    #     cumulated_counter = 0.0
    #     idx_at_best_succ = 0
    #     successes = []
    #     losses = []
    #
    #     # start training
    #     for epoch in range(0, self.cfg.adaptation.n_epochs + 1):
    #
    #         t0 = time.time()
    #         if epoch > 0 or (self.cfg.pretrain):  # update
    #             self.policy.train()
    #             training_loss = 0.0
    #             for (idx, data) in enumerate(train_dataloader):
    #                 loss = self.observe(data)
    #                 training_loss += loss
    #             training_loss /= len(train_dataloader)
    #         else:  # just evaluate the zero-shot performance on 0-th epoch
    #             training_loss = 0.0
    #             #
    #             for (idx, data) in enumerate(train_dataloader):
    #                 loss = self.eval_observe(data)
    #                 training_loss += loss
    #             training_loss /= len(train_dataloader)
    #         t1 = time.time()
    #
    #         print(
    #             f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1 - t0) / 60:4.2f}"
    #         )
    #
    #         if epoch % self.cfg.adaptation.eval_every == 0:  # evaluate BC loss
    #             t0 = time.time()
    #             model_checkpoint_name_ep = os.path.join(
    #                 self.experiment_dir, f"lora_model_ep{epoch}.pth"
    #             )
    #
    #             # only save the lora parameters
    #             torch.save({
    #                 "state_dict": lora.lora_state_dict(self.policy, bias=which_bias_train),
    #                 "cfg": self.cfg,
    #             }, model_checkpoint_name_ep)
    #
    #             losses.append(training_loss)
    #
    #             # for multitask learning, we provide an option whether to evaluate
    #             # the agent once every eval_every epochs on all tasks, note that
    #             # this can be quite computationally expensive. Nevertheless, we
    #             # save the checkpoints, so users can always evaluate afterwards.
    #             if self.cfg.adaptation.eval:
    #                 self.policy.eval()
    #
    #                 success_rates = evaluate_pretrain_multitask_training_success(
    #                     self.cfg, self, benchmark, adapt_task
    #                 )
    #                 success_rate = np.mean(success_rates)
    #
    #                 successes.append(success_rate)
    #
    #                 if prev_success_rate < success_rate and (not self.cfg.pretrain):
    #                     # torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg)
    #                     torch.save({
    #                         "state_dict": lora.lora_state_dict(self.policy, bias=which_bias_train),
    #                         "cfg": self.cfg,
    #                     }, model_checkpoint_name)
    #                     prev_success_rate = success_rate
    #                     idx_at_best_succ = len(losses) - 1
    #
    #                 t1 = time.time()
    #
    #                 cumulated_counter += 1.0
    #                 ci = confidence_interval(success_rate, self.cfg.eval.n_eval)
    #                 tmp_successes = np.array(successes)
    #                 tmp_successes[idx_at_best_succ:] = successes[idx_at_best_succ]
    #
    #                 if self.cfg.adaptation.eval_in_train:
    #                     print(
    #                         f"[info] Epoch: {epoch:3d} | succ: {success_rate:4.2f} ± {ci:4.2f} | best succ: {prev_success_rate} "
    #                         + f"| succ. AoC {tmp_successes.sum() / cumulated_counter:4.2f} | time: {(t1 - t0) / 60:4.2f}",
    #                         flush=True,
    #                     )
    #
    #         if self.scheduler is not None and epoch > 0:
    #             self.scheduler.step()
    #
    #     # # eval the model in the envs
    #     # final_success_rates = evaluate_pretrain_multitask_training_success(
    #     #     self.cfg, self, benchmark, adapt_task
    #     # )
    #     # final_success_rate = np.mean(final_success_rates)
    #     # print(self.cfg.experiment_dir)
    #     # print('Final success rate:', final_success_rates)
    #     # print('Final avg success rate:', final_success_rate)
    #
    #     # load the best policy if there is any
    #     if self.cfg.lifelong.eval_in_train:
    #         self.policy.load_state_dict(torch_load_model(model_checkpoint_name)[0])
    #     self.end_task(concat_dataset, -1, benchmark)
    #
    #     # return the metrics regarding forward transfer
    #     losses = np.array(losses)
    #     successes = np.array(successes)
    #     auc_checkpoint_name = os.path.join(
    #         self.experiment_dir, f"multitask_auc.log"
    #     )
    #     torch.save(
    #         {
    #             "success": successes,
    #             "loss": losses,
    #         },
    #         auc_checkpoint_name,
    #     )
    #
    #     if self.cfg.lifelong.eval_in_train:
    #         loss_at_best_succ = losses[idx_at_best_succ]
    #         success_at_best_succ = successes[idx_at_best_succ]
    #         losses[idx_at_best_succ:] = loss_at_best_succ
    #         successes[idx_at_best_succ:] = success_at_best_succ
