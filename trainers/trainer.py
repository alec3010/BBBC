import numpy as np
import torch
import math
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR

from utils import helpers as h

class Trainer():
    def __init__(self, env_name, configs) -> None:
        self.config = configs
        self.get_params()
        self.env_name = env_name
        self.load_dataset_idx()
        print("Configs are: ","\n" , self.config)
        
        # self.train_x, self.train_y, self.val_x, self.val_y  = h.generate_sin_dataset(50000, self.split)
        self.writer = SummaryWriter(log_dir = "./tensorboard")
        self.reset_results()

    def init_optimizer(self):
        self.opt = torch.optim.Adam(self.model.parameters(),lr=self.lr)# adam optimization
        self.scheduler = ExponentialLR(optimizer=self.opt, gamma=self.gamma, verbose=True)
        self.mse = torch.nn.MSELoss()# MSE loss

    def get_params(self):
        self.prev_acs = self.config['prev_acs']
        self.lr = self.config['learning_rate']
        self.gamma = self.config['gamma']
        self.epochs = self.config['epochs']
        self.eval_int = self.config['eval_interval']
        self.batch_size = self.config['batch_size']
        self.seq_length = self.config['seq_length']
        self.split = self.config['split']
        self.k = self.config['k']
        self.loss_weights = self.config['loss_weights']
        self.hidden_dim = self.config['hidden_dim']
        self.decoder_hidden = self.config['decoder_hidden']
        self.belief_dim = self.config['belief_dim']
        self.policy_hidden = self.config['policy_hidden']
        self.acs_encoding_dim = self.config['acs_encoding_dim']
        self.ae_state_dict = self.config['ae_state_dict']
        self.policy_state_dict = self.config['policy_state_dict']

    def load_dataset_idx(self):
        dataset_idx = h.get_params("configs/dataset_index.yaml")
        entry = dataset_idx[self.env_name]
        self.db_path_belief = entry['db_belief']
        self.db_path_policy = entry['db_policy']
        self.obs_dim = len(entry['obs_dim'])
        self.acs_dim = entry['acs_dim']
        self.idx_list = entry['obs_dim']

    def reset_results(self):
        self.result_dict = {}
        self.result_dict['env'] = self.env_name
        self.result_dict['learning_params'] = self.config
        
        self.result_dict['train_loss'] = {}
        self.result_dict['train_loss']['epoch'] = []
        self.result_dict['train_loss']['value'] = []
        self.result_dict['val_loss'] = {}
        self.result_dict['val_loss']['epoch'] = []
        self.result_dict['val_loss']['value'] = []
        self.result_dict['reward'] = 0
    
    def get_results(self):
        print(self.result_dict['learning_params'])
        return self.result_dict

    
        
    