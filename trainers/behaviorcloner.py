import os
import pickle
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR

from utils.pytorchtools import EarlyStopping
from utils import helpers as h
from utils.data_loader import DataLoader
from eval_env import EvaluationEnvironment
import models.models as m

class BehaviorCloner():

    def __init__(self, env_name, configs) -> None:
        print("Creating Behavior cloner Object")
        
        self.config = configs
        self.get_params()
        self.env_name = env_name
        self.load_dataset_idx()
        print("Configs are: ","\n" , self.config)
        self.agent = m.model_factory(configs['network_arch'], obs_dim=self.obs_dim, acs_dim=self.acs_dim, configs=configs)
        
        self.init_optimizer()
        
        self.train_x, self.train_y, self.val_x, self.val_y  = h.train_val_split(self.db_path, self.split)
        self.writer = SummaryWriter(log_dir = "./tensorboard")
        self.model_dir = "./models"
        
        self.reset_results()

    def train_policy(self):
        stopper = EarlyStopping(patience=100, verbose=True)
        print("... train model")
        self.agent.train()
        steps = 0
        for epoch in range(self.epochs):
            
            loader = DataLoader(self.train_x, 
                                   self.train_y, 
                                   self.seq_length, 
                                   self.idx_list,
                                   self.network_arch)
            n_iters = 0
            train_loss = 0.0
            
            for (x, y) in loader:
                if self.network_arch == "RNNFF":
                    outputs, hidden = self.agent(x) # agent, pytorch
                elif self.network_arch == "FF":
                    outputs = self.agent(x) # agent, pytorch
                loss = self.criterion(outputs.float(), y.float()) # mse loss
                self.optimizer.zero_grad() # reset weights
                loss.backward() # backprop
                self.optimizer.step() # adam optim, gradient update
                
                train_loss+=loss.item() * x.size(0)
                
                n_iters+=1
                steps += 1
            if (epoch+1)%50 == 0 and epoch != 0:
                self.scheduler.step()
            
            self.writer.add_scalar("Loss/train", (train_loss/len(self.train_x)), epoch + 1)   
            #print("average train trajectory loss in epoch " + str(epoch + 1) + ": " + str(train_loss / n_iters))
            self.result_dict['train_loss']['epoch'].append(epoch + 1)
            self.result_dict['train_loss']['value'].append(train_loss/len(self.train_x))
            
            if (epoch+1)%self.eval_int == 0 and epoch != 0:
                
                val_loss = self.eval_policy(epoch)
                self.writer.add_scalar("Loss/val", (val_loss), epoch + 1)   
                stopper(val_loss, self.agent)
                if stopper.early_stop:
                    print("Early stop")
                    break

            
            
            
        
        reward = self.eval_on_ss()
        # reward = self.eval_on_env()
        #print('Reward on Environment: %f' % reward )

    def eval_policy(self, epoch):
        
        loader = DataLoader(self.val_x,
                               self.val_y,
                               self.seq_length, 
                               self.idx_list,
                               self.network_arch)
        
        valid_loss= 0.0
        self.agent.eval()
        for (x, y) in loader:
            
            if self.network_arch == "RNNFF":
                outputs, hidden, _ = self.agent(x) # agent, pytorch
            elif self.network_arch == "FF":
                outputs = self.agent(x) # agent, pytorch
            loss = self.criterion(outputs, y) # mse loss
            valid_loss += loss.item() * x.size(0)
            

        avg_loss = valid_loss/len(loader)
        print('Validation Loss in Epoch {}: {}'.format(epoch+1, avg_loss))
        self.result_dict['val_loss']['epoch'].append(epoch + 1)
        self.result_dict['val_loss']['value'].append(avg_loss)
            #print(f'valid set accuracy: {valid_acc}')

        self.agent.train()
        return avg_loss
        
   

    

    
    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(self.agent.parameters(),lr=self.lr)# adam optimization
        self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=self.gamma, verbose=True)
        self.criterion = torch.nn.MSELoss()# MSE loss

    def save_model(self):
        torch.save(self.agent.state_dict(), os.path.join(model_dir,"InvertedPendulum.pkl"))

    def get_params(self):
        self.lr = self.config['learning_rate']
        self.gamma = self.config['gamma']
        self.process_model = self.config['process_model']
        self.network_arch = self.config['network_arch']
        self.epochs = self.config['epochs']
        self.eval_int = self.config['eval_interval']
        self.shuffle = self.config['shuffle']
        self.batch_size = self.config['batch_size']
        self.seq_length = self.config['seq_length']
        self.split = self.config['split']

    def load_dataset_idx(self):
        dataset_idx = h.get_params("configs/dataset_index.yaml")
        entry = dataset_idx[self.env_name]
        self.db_path = entry['db']
        self.obs_dim = len(entry['obs_dim'][self.process_model])
        self.acs_dim = entry['acs_dim']
        self.idx_list = entry['obs_dim'][self.process_model]

    def eval_on_env(self):
        eval_env = EvaluationEnvironment(self.agent, self.env_name, self.idx_list, self.config)
        avg_reward = eval_env.eval_mjc()
        
        return avg_reward

    def eval_on_ss(self):
        eval_env = EvaluationEnvironment(self.agent, self.env_name, self.idx_list, self.config)
        eval_env.eval_ss()




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

    

    

    

    
        
        

    
        

    



