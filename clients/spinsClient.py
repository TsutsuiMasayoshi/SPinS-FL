import os
import sys
sys.path.append(os.curdir)
from train import train, test
from constants import CLIENTNUM_GROUP, GROUP_NUM

import numpy as np
import torch
from collections import OrderedDict
import flwr as fl

class SpinsClient(fl.client.NumPyClient):
    def __init__(self, model, optimizer, criterion, device, train_loaders, test_loader, args) -> None:
        torch.backends.cudnn.benchmark  = True
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.args = args
        self.round_ctr = 0
        self.score_without_bn_idx = []
        self.top_untracked_indices_list = [np.array([],dtype=int) for _ in range(6+3)] # indices of top half untracked scores per layer
        self.bottom_untracked_indices_list = [np.array([],dtype=int) for _ in range(6+3)] # bottom half

        idx = 0
        for name in self.model.state_dict().keys():
            if 'scores' in name or ('bn' in name and 'running' in name):
                if 'scores' in name:
                    self.score_without_bn_idx.append(idx)
                idx += 1

    def get_parameters(self, config):
        score_bn_list = []
        for name, val in self.model.state_dict().items():
            if 'scores' in name or ('bn' in name and 'running' in name):
                score_bn_list.append(val.cpu().numpy())
        return score_bn_list
        
    def globalPinning(self, score_bn_list):

        layer_idx = 0
        for scoreidx in self.score_without_bn_idx: #killing global score
            v = score_bn_list[scoreidx]
            v_flatten = v.flatten()

            dead_indices = np.concatenate([self.top_untracked_indices_list[layer_idx], self.bottom_untracked_indices_list[layer_idx]], 0)

            all_indices = np.arange(len(v_flatten))
            live_mask = np.ones(all_indices.size, dtype=bool)
            live_mask[dead_indices] = False
            live_indices = all_indices[live_mask]
            v_alive_flatten = v_flatten[live_mask]
            abs_lives_ranking = np.argsort(np.abs(v_alive_flatten))
            sorted_live_indices = live_indices[abs_lives_ranking] # sorted by their score ranks

            if live_indices.size // 2 - int((self.args.globalPinRate / 2) * v.size) > v.size // 2 - int((self.args.localPinRate / 2) * v.size): # do not invade scores not localPinned yet.
                bottomkilling_end = int((self.args.globalPinRate / 2) * v.size)
                topkilling_start = sorted_live_indices.size - bottomkilling_end # the same number of scores are killed from top and bottom
            else:
                bottomkilling_end = sorted_live_indices.size // 2 - int(((1.0 - self.args.localPinRate) / 2) * v.size) # do not invade scores not localPinned yet.
                topkilling_start = sorted_live_indices.size - bottomkilling_end

            bottom_killed_indices = sorted_live_indices[:bottomkilling_end]
            top_killed_indices = sorted_live_indices[topkilling_start:]

            new_bottom_untracked = np.concatenate([self.bottom_untracked_indices_list[layer_idx], bottom_killed_indices], 0)
            new_top_untracked = np.concatenate([self.top_untracked_indices_list[layer_idx], top_killed_indices], 0)
            self.bottom_untracked_indices_list[layer_idx] = new_bottom_untracked
            self.top_untracked_indices_list[layer_idx] = new_top_untracked

            layer_idx += 1


    def set_parameters(self, score_bn_list):
        top_indices_list = []
        unlocked_indices_list = []
        bottom_indices_list = []

        layer_idx = 0
        for scoreidx in self.score_without_bn_idx: # pinning local score
            v = score_bn_list[scoreidx]

            ### global pinning support
            living_mask = np.ones(v.size, dtype=bool)
            all_indices = np.arange(v.size)

            living_mask[self.top_untracked_indices_list[layer_idx]] = False
            living_mask[self.bottom_untracked_indices_list[layer_idx]] = False

            living_v_flatten = v.flatten()[living_mask]
            live_indices = all_indices[living_mask]
                
            abs_livingscores_ranking = np.argsort(np.abs(living_v_flatten))
            sorted_live_indices = live_indices[abs_livingscores_ranking] # sorted by their score ranks

            bottom_locked_end = living_v_flatten.size // 2 - int((1.0 - self.args.localPinRate) * v.size) // 2 # the number of scores to pin
            top_locked_start = living_v_flatten.size - bottom_locked_end

            bottom_locked_indices = np.concatenate([self.bottom_untracked_indices_list[layer_idx], sorted_live_indices[:bottom_locked_end]], 0)
            unlocked_indices = sorted_live_indices[bottom_locked_end:top_locked_start]
            top_locked_indices = np.concatenate([self.top_untracked_indices_list[layer_idx], sorted_live_indices[top_locked_start:]], 0)

            top_indices_list.append(top_locked_indices)
            unlocked_indices_list.append(unlocked_indices)
            bottom_indices_list.append(bottom_locked_indices)
            layer_idx += 1

        parameters = []
        i = 0
        top_itr = 0
        unlocked_itr = 0
        bottom_itr = 0

        for name, val in self.model.state_dict().items():
            if 'scores' in name or ('bn' in name and 'running' in name):
                parameters.append(score_bn_list[i])
                i += 1
            elif 'top_locked' in name:
                parameters.append(top_indices_list[top_itr])
                top_itr += 1
            elif 'unlocked' in name:
                parameters.append(unlocked_indices_list[unlocked_itr])
                unlocked_itr += 1
            elif 'bottom_locked' in name:
                parameters.append(bottom_indices_list[bottom_itr])
                bottom_itr += 1
            else:
                parameters.append(val)

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        if self.round_ctr % 10 == 0 and self.round_ctr != 0:
            self.globalPinning(parameters)

        self.set_parameters(parameters)

        if not self.args.is_iid:
            perm = torch.randperm(CLIENTNUM_GROUP)
            selected_from_each = perm[self.args.client_id]
            train_loader = self.train_loaders[selected_from_each]
        else:
            virtids_participate = np.random.randint(0, CLIENTNUM_GROUP-1, GROUP_NUM)
            train_loader = self.train_loaders[virtids_participate[self.args.client_id]]

        train(self.model, self.device, train_loader, self.optimizer, self.criterion, self.args.epochs)

        self.round_ctr += 1

        return self.get_parameters(config), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.device, self.criterion, self.test_loader)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}