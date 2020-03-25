import os
from abc import abstractmethod

import torch
import torch.optim as optim
import torch.nn as nn

from bpe.functional.utils import TrainClock


class BaseAgent(object):
    def __init__(self, config, net):
        self.log_dir = config.log_dir
        self.model_dir = config.model_dir
        self.net = net
        self.clock = TrainClock()
        self.device = config.device
        self.use_footvel_loss = config.use_footvel_loss

        # set loss function
        self.mse = nn.MSELoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=config.triplet_margin)
        self.triplet_weight = config.triplet_weight
        self.foot_idx = config.foot_idx
        self.footvel_loss_weight = config.footvel_loss_weight

        # set optimizer
        if config.L2regular:
           self.optimizer = optim.Adam(self.net.parameters(), config.lr, weight_decay=config.weight_decay)
        else:
            self.optimizer = optim.Adam(self.net.parameters(), config.lr)
            
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.lr_decay_rate)

    def save_network(self, name=None):
        if name is None:
            save_path = os.path.join(self.model_dir, "model_epoch{}.pth".format(self.clock.epoch))
        else:
            save_path = os.path.join(self.model_dir, name)
        torch.save(self.net.module.state_dict(), save_path)

    def load_network(self, epoch):
        load_path = os.path.join(self.model_dir, "model_epoch{}.pth".format(epoch))
        state_dict = torch.load(load_path)
        self.net.load_state_dict(state_dict)
        self.net.to(self.device)

    @abstractmethod
    def forward(self, data):
        pass

    def update_network(self, loss_dict):
        loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        self.scheduler.step(self.clock.lr_step)

    def train_func(self, data):
        self.net.train()
        losses = self.forward(data)
        self.update_network(losses)
        return losses

    def val_func(self, data):
        self.net.eval()
        with torch.no_grad():
            losses = self.forward(data)
        return losses
