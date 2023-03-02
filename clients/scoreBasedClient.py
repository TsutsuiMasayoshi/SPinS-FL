import os
import sys
sys.path.append(os.curdir)
from train import train, test
from constants import CLIENTNUM_GROUP, GROUP_NUM

import numpy as np
import torch
from collections import OrderedDict
import flwr as fl

class ScoreBasedClient(fl.client.NumPyClient):
    def __init__(self, model, optimizer, criterion, device, train_loaders, test_loader, trafficTracker, args) -> None:
        torch.backends.cudnn.benchmark  = True
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.trafficTracker = trafficTracker
        self.args = args
        self.round_ctr = 0

    def get_parameters(self, config):
        layersToComm = []
        for name, val in self.model.state_dict().items():
            if 'scores' in name or ('bn' in name and 'running' in name):
                layersToComm.append(val.cpu().numpy())
        if self.args.client_id == 0: # one communication recorder is enough
            self.trafficTracker.send(sum([l.size for l in layersToComm]))
        return layersToComm

    def set_parameters(self, layersToComm):
        if self.args.client_id == 0: # one communication recorder is enough
            self.trafficTracker.load(sum([l.size for l in layersToComm]))
        parameters = []
        i = 0
        # replace only score vals with those averaged by server
        for name, val in self.model.state_dict().items():
            if 'scores' in name or ('bn' in name and 'running' in name):
                parameters.append(layersToComm[i])
                i += 1
            else:
                parameters.append(val)
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
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

        return self.get_parameters(config), len(train_loader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.device, self.criterion, self.test_loader)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}