import os
import pickle
import numpy as np
import torch
import math
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from trainers.trainer import Trainer

from utils.pytorchtools import EarlyStopping
from utils import helpers as h
from utils.data_loader import DataLoader
from eval_env import EvaluationEnvironment
from models.GRUVAE import GRUVAE
from models.FF import FF
from models.FFDO import FFDO


class PolicyTrainer(Trainer):

    def __init__(self, env_name, configs) -> None:
        super(PolicyTrainer,self).__init__(env_name=env_name, configs=configs)
        print("Creating PolicyTrainer Object")
        
        self.vae = GRUVAE(self.obs_dim,self.acs_dim, self.belief_dim, self.hidden_dim, self.decoder_hidden).cuda()
        self.vae.load_state_dict(torch.load(self.vae_state_dict))
        self.model = FFDO(self.acs_dim, self.belief_dim, self.policy_hidden).cuda()
        self.init_optimizer()
            
    def train(self):
        stopper = EarlyStopping(patience=10, verbose=False)
        print("... train policy model")
        self.model.train()
        k = self.k
        for epoch in range(self.epochs):
            
            loader = DataLoader(self.train_x, 
                                self.train_y, 
                                self.seq_length, 
                                self.idx_list,
                                self.network_arch)
            n_iters = 0
            train_loss = 0.0
            
            for (x, y) in loader:
                
                _, mu_s, _ = self.vae(x[k:-k]) # vae, pytorch
                acs = self.model(mu_s.detach())
                loss = self.mse(acs, y[k:-k])
                
                self.opt.zero_grad() # reset weights
                loss.backward() # backprop
                self.opt.step() # adam optim, gradient update
                train_loss+=loss.item()
                n_iters+=1
            if (epoch+1)%50 == 0 and epoch != 0: 
                self.scheduler.step()
            
            self.writer.add_scalar("LossPOL/TRAIN", (train_loss/n_iters), epoch + 1)   
            
            if (epoch+1)%self.eval_int == 0 and epoch != 0:
                
                val_loss = self.eval(epoch)
                self.writer.add_scalar("LossPOL/VAL", (val_loss), epoch + 1)   
                stopper(val_loss, self.vae)
                if stopper.early_stop:
                    print("Early stop")
                    break

        self.eval_on_ss()
        torch.save(self.model.state_dict(), self.policy_state_dict)

    def eval(self, epoch):
        loader = DataLoader(self.val_x,
                               self.val_y,
                               self.seq_length, 
                               self.idx_list,
                               self.network_arch)
        
        valid_loss = 0.0
        self.vae.eval()
        self.model.eval()
        k = self.k
        n_iters = 0
        for (x, y) in loader:
            
            pred, mu_s, sigma_s, _ = self.vae(x[k:-k]) # vae, pytorch
            acs = self.model(mu_s.detach())
            loss = self.mse(acs, y[k:-k])
            valid_loss += loss.item()
            n_iters += 1

        avg_loss = round(valid_loss/n_iters, 4)
        epoch_str = h.epoch_str(epoch)
        print('{}||POL Loss: {}'.format(epoch_str, avg_loss))
    

        self.model.train()
        self.vae.train()
        return avg_loss

    def eval_on_env(self):
        eval_env = EvaluationEnvironment(self.vae, self.model, self.env_name, self.idx_list, self.config)
        avg_reward = eval_env.eval_mjc()
        
        return avg_reward

    def eval_on_ss(self):
        eval_env = EvaluationEnvironment(self.vae, self.model, self.env_name, self.idx_list, self.config)
        eval_env.eval_ss()


