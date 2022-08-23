import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from model import pyTorchModel


class BehaviorCloner():

    def __init__(self, belief_dim, action_dim, obs_dim, actor_hidden_dim, lr) -> None:
        self.optimizer = torch.optim.Adam(self.agent.parameters(),lr=lr)# adam optimization
        self.criterion = torch.nn.MSELoss()# MSE loss
        self.writer = SummaryWriter(log_dir = "./tensorboard")
        

    def train_val_split(self, datasets_dir="./data", frac = 0.1)
        print("... read data")
        data_file = os.path.join(datasets_dir, 'sac_demo_InvertedPendulum_final_pomdp.pickle')
    
        f = open(data_file,'rb')
        data = pickle.load(f)

        # split data into training and validation set
        n_samples = len(data)
        train = data[:int((1-frac) * n_samples)]
        val = data[int((1-frac) * n_samples):]
        return train, val

    



