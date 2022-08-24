import os
import pickle
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import models as m
from utils import helpers as h

class BehaviorCloner():

    def __init__(self, env_name) -> None:
        self.env_name = env_name
        self.get_params()
        self.load_dataset_idx()
        self.writer = SummaryWriter(log_dir = "./tensorboard")
        self.model_dir = "./models"
        
        self.frac = 0.1

    def train_val_split(self):
        print("... read data")
        data_file = os.path.join(self.db_path)
    
         
        f = open(data_file,'rb')
        data = pickle.load(f)
        n_samples = len(data)
        idx = int((1-self.frac) * n_samples)
        # split data into training and validation set
        
        train = data[:idx]
        val = data[idx:]
        return train, val
    
    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(self.agent.parameters(),lr=self.lr)# adam optimization
        self.criterion = torch.nn.MSELoss()# MSE loss

    def save_model(self):

        torch.save(self.agent.state_dict(), os.path.join(model_dir,"InvertedPendulum.pkl"))

    def get_params(self):
        self.config = h.get_params("./configs/learning_params.yaml")
        self.lr = self.config['learning_rate']
        self.process_model = self.config['process_model']
        self.network_arch = self.config['network_arch']
        self.epochs = self.config['epochs']
        self.eval_int = self.config['eval_interval']
        self.shuffle = self.config['shuffle']
        self.batch_size = self.config['batch_size']

    def load_dataset_idx(self):
        dataset_idx = h.get_params("configs/dataset_index.yaml")
        entry = dataset_idx[self.env_name]
        print(entry)
        self.db_path = entry['db'][self.process_model]
        self.obs_dim = entry['obs_dim'][self.process_model]
        self.acs_dim = entry['acs_dim']
        
        

    
        

    



