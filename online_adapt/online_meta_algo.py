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
from lora_parts.policy import *
from torch import autograd
from utils.online_adapt_utils import clone_module, update_module
import json
from libero.lifelong.utils import NpEncoder
from omegaconf import OmegaConf
import yaml
from easydict import EasyDict
from copy import deepcopy
from torch.utils.data import Dataset


class MetaDataset(Dataset):
    def __init__(self, dataset, support_num=4, query_num=1, search_region=10):
        self.meta_dataset = dataset
        self.support_num = support_num
        self.query_num = query_num
        self.search_region = search_region
        assert dataset.n_demos > 1, "not enough demos"

        self.index_min = min(self.meta_dataset.sequence_dataset._demo_id_to_demo_length.values())

    def __len__(self):
        return self.index_min

    def __getitem__(self, ind):
        demo_ids = random.sample(self.meta_dataset.sequence_dataset.demos, 2)

        support_demo = demo_ids[0]
        query_demo = demo_ids[1]

        support_demo_start = self.meta_dataset.sequence_dataset._demo_id_to_start_indices[support_demo]
        query_demo_start = self.meta_dataset.sequence_dataset._demo_id_to_start_indices[query_demo]

        support_demo_len = self.meta_dataset.sequence_dataset._demo_id_to_demo_length[support_demo]
        query_demo_len = self.meta_dataset.sequence_dataset._demo_id_to_demo_length[query_demo]

        query_id = query_demo_start + ind

        support_center_id = support_demo_start + ind
        sampled_support_low = max(support_demo_start, support_center_id - self.search_region)
        sampled_support_high = min(support_center_id + self.search_region, support_demo_start + support_demo_len)
        sampled_support_id = random.sample(range(sampled_support_low, sampled_support_high), k=self.support_num - 1)
        sampled_support_id.append(support_center_id)

        obs_keys = ['agentview_rgb', 'eye_in_hand_rgb', 'gripper_states', 'joint_states']
        support_dataset = {}
        query_dataset = {}

        def init_dataset_dict(dataset_dict, obs_keys):
            dataset_dict['actions'] = []
            dataset_dict['task_emb'] = []
            dataset_dict['obs'] = {}
            for k in obs_keys:
                dataset_dict['obs'][k] = []
            return dataset_dict

        def stack_dataset(dataset_dict, obs_keys):
            dataset_dict['actions'] = np.stack(dataset_dict['actions'], axis=0)
            dataset_dict['task_emb'] = np.stack(dataset_dict['task_emb'], axis=0)
            for k in obs_keys:
                dataset_dict['obs'][k] = np.stack(dataset_dict['obs'][k], axis=0)
            return dataset_dict

        support_dataset = init_dataset_dict(support_dataset, obs_keys)
        query_dataset = init_dataset_dict(query_dataset, obs_keys)

        for i in sampled_support_id:
            data = self.meta_dataset.__getitem__(i)
            support_dataset['actions'].append(data['actions'])
            for k in obs_keys:
                support_dataset['obs'][k].append(data['obs'][k])
            support_dataset['task_emb'].append(data['task_emb'])

        query_data = self.meta_dataset.__getitem__(query_id)
        query_dataset['actions'].append(query_data['actions'])
        for k in obs_keys:
            query_dataset['obs'][k].append(query_data['obs'][k])
        query_dataset['task_emb'].append(query_data['task_emb'])

        support_dataset = stack_dataset(support_dataset, obs_keys)
        query_dataset = stack_dataset(query_dataset, obs_keys)

        return support_dataset, query_dataset


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

        self.meta_optimizer = eval(self.cfg.adaptation.optim_name)(
            list(self.policy.parameters()) + list(self.meta_update_inner_lr.values()),
            **self.cfg.adaptation.meta_optim_kwargs
        )

    def online_adapt(self, pre_train_dataset, post_adaptation_dataset):
        self.meta_lora_params = deepcopy(lora.lora_state_dict(self.policy, bias=self.cfg.adaptation.bias_training_type))
        for task in range(len(post_adaptation_dataset)):
            self.start_task(task + self.cfg.adaptation.post_adaptation_start_id)
            existing_dataset = pre_train_dataset + post_adaptation_dataset[:task + 1]
            existing_dataset = [MetaDataset(ds) for ds in existing_dataset]
            self.meta_update(existing_dataset=existing_dataset)
            self.update_procedure(task_specific_dataset=post_adaptation_dataset[task])
            self.load_meta_lora_params()  # load the lora params which are before the fine tunning

    def meta_update(self, existing_dataset):

        iter_dl_list = self.turn_to_iters(existing_dataset, batch_size=1)

        print('--------- Meta Learning Phase ---------------')
        for ep in range(self.cfg.adaptation.meta_update_epochs):
            ep_query_loss = []

            for i in range(len(iter_dl_list)):
                try:
                    data = next(iter_dl_list[i])
                except:
                    iter_dl_list[i] = iter(DataLoader(existing_dataset[i], batch_size=1, shuffle=True,
                                                      num_workers=self.cfg.adaptation.num_workers, drop_last=True))
                    data = next(iter_dl_list[i])
                support_data, query_data = data
                adapted_policy_net = self._meta_inner_step(support_data)
                loss = self.meta_val(adapted_policy_net, query_data)
                ep_query_loss.append(loss)

            # TODO logging the loss
            self.meta_optimizer.zero_grad()
            mean_loss = torch.mean(torch.stack(ep_query_loss))
            mean_loss.backward()
            self.meta_optimizer.step()
            if ep % self.cfg.adaptation.meta_display_interval == 0:
                print(f'epoch: {ep}, meta query loss: {mean_loss.item()}, current task: {self.current_task}')

        print('##################################')
        self.save_meta_lora_params()

    def update_procedure(self, task_specific_dataset):

        train_dataloader = DataLoader(
            task_specific_dataset, batch_size=self.cfg.adaptation.update_procedure_batch_size, shuffle=True,
            num_workers=self.cfg.adaptation.num_workers)

        print('--------- Update Procedure Phase ---------------')
        for epoch in range(1, self.cfg.adaptation.n_epochs + 1):

            t0 = time.time()

            self.policy.train()
            training_loss = 0.0
            for (idx, data) in enumerate(train_dataloader):
                loss = self.observe(data)
                training_loss += loss
            training_loss /= len(train_dataloader)

            t1 = time.time()

            # print(
            #     f"[info] Task: {self.current_task}| Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1 - t0) / 60:4.2f}"
            # )
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
        temp = deepcopy(lora.lora_state_dict(self.policy, bias=self.cfg.adaptation.bias_training_type))
        path = os.path.join(self.cfg.experiment_dir, f'meta_params_task_{self.current_task}.pth')
        torch.save(temp, path)

    def load_meta_lora_params(self):
        self.policy.load_state_dict(self.meta_lora_params, strict=False)

    def turn_to_iters(self, datasets, batch_size):
        iters_list = []

        for i in range(len(datasets)):
            temp_dl = DataLoader(datasets[i], batch_size=batch_size, shuffle=True,
                                 num_workers=self.cfg.adaptation.num_workers, drop_last=True)
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
        query_data['actions'] = query_data['actions'][0]
        query_data['task_emb'] = query_data['task_emb'][0]
        keys = list(query_data['obs'].keys())
        for k in keys:
            query_data['obs'][k] = query_data['obs'][k][0]
        data = self.map_tensor_to_device(query_data)
        # self.meta_optimizer.zero_grad()

        loss = self.loss_scale * adapted_policy_net.compute_loss(data)
        return loss

        # loss.backward()
        #
        # self.meta_optimizer.step()

    def _meta_inner_step(self, data):
        # squeeze the axis 0 of the data
        data['actions'] = data['actions'][0]
        data['task_emb'] = data['task_emb'][0]
        keys = list(data['obs'].keys())
        for k in keys:
            data['obs'][k] = data['obs'][k][0]

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
                    p.update = - 0.4 * g
        target_policy_net = update_module(target_policy_net)

        return target_policy_net

    def start_task(self, task):
        """
        What the algorithm does at the beginning of learning each lifelong task.
        """
        self.current_task = task

        # initialize the optimizer and scheduler
        self.optimizer = eval(self.cfg.adaptation.optim_name)(
            self.policy.parameters(), **self.cfg.adaptation.optim_kwargs
        )
