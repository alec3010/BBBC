import os
import pickle
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils import helpers as h
from eval_env import EvaluationEnvironment

class BehaviorCloner():

    def __init__(self, env_name, configs) -> None:
        print("Creating Behavior cloner Object")
        
        self.config = configs
        print("Configs are: ","\n" , self.config)

        self.env_name = env_name
        self.get_params()
        self.load_dataset_idx()
        self.writer = SummaryWriter(log_dir = "./tensorboard")
        self.model_dir = "./models"
        
        self.frac = 0.1
        self.reset_results()
        
   

    
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
        self.db_path = entry['db']
        self.obs_dim = len(entry['obs_dim'][self.process_model])
        self.acs_dim = entry['acs_dim']
        self.idx_list = entry['obs_dim'][self.process_model]

    def eval_on_env(self):
        eval_env = EvaluationEnvironment(self.agent, self.env_name, self.idx_list, self.config)
        avg_reward = eval_env.eval()

        return avg_reward

    def occlusion(self, point):
        if self.process_model == "pomdp":
            _ = []
            for idx in self.idx_list:

                _.append(point['obs'][idx].item())
            
            obs = torch.cuda.FloatTensor(_)
            #print(obs.size())
        if self.process_model == "mdp":
            #print("mdp")
            obs = torch.cuda.FloatTensor(point['obs']).float()
        return obs

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

    

    
        
        

    
        

    



