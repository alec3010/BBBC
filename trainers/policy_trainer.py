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
from models.auto_encoder import AutoEncoder
from models.FF import FF


class PolicyTrainer(Trainer):

    def __init__(self, env_name, configs) -> None:
        super(PolicyTrainer,self).__init__(env_name=env_name, configs=configs)
        print("Creating PolicyTrainer Object")
        self.train_x, self.train_y, self.val_x, self.val_y  = h.train_val_split(self.db_path_policy, self.split, self.idx_list)
        if self.prev_acs:
            self.ae = AutoEncoder(self.obs_dim + self.acs_dim, self.acs_dim, self.acs_encoding_dim,self.obs_dim, self.belief_dim, self.hidden_dim, self.decoder_hidden, self.k, self.prev_acs).cuda()
        else:
            self.ae = AutoEncoder(self.obs_dim, self.acs_dim, self.acs_encoding_dim, self.obs_dim, self.belief_dim, self.hidden_dim, self.decoder_hidden, self.k, self.prev_acs).cuda()
        self.ae.load_state_dict(torch.load(self.ae_state_dict))
        self.ae.test()
        self.model = FF(self.acs_dim, self.belief_dim, self.policy_hidden).cuda()
        self.init_optimizer()
            
    def train(self):
        self.hidden = None
        stopper = EarlyStopping(patience=50, verbose=False)
        print("... train policy model")
        self.model.train()
        
        k = self.k
        for epoch in range(self.epochs):
      
            
            n_iters = 0
            train_loss = 0.0
            
            for (x, y) in zip(self.train_x, self.train_y):
                
                self.ae.init_hidden()
                y_pad = h.zero_pad(y)
                y_prev = y_pad[:-k-1]
                mu, mu_list = self.ae(x[:-k], y_prev)
                acs = self.model(mu.detach())
                loss = self.mse(acs, y[:-k])
                
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
                stopper(val_loss, self.model)
                if stopper.early_stop:
                    print("Early stop")
                    break

        self.eval_on_ss()
        torch.save(self.model.state_dict(), self.policy_state_dict)

    def eval(self, epoch):
        valid_loss = 0.0
       
        self.model.eval()
        self.ae.eval()
        k = self.k
        n_iters = 0
        for (x, y) in zip(self.val_x, self.val_y):
            # future_acs = torch.squeeze(h.tm1_tpkm1(y, k), 2)
            # past_acs = torch.squeeze(h.tmkm1_tm1(y, k), 2)
            y_pad = h.zero_pad(y)
            y_prev = y_pad[:-k-1]
            self.ae.init_hidden()
            mu, mu_list = self.ae(x[:-k], y_prev)
            
            acs = self.model(mu.detach())
            loss = self.mse(acs, y[:-k])
            valid_loss += loss.item()
            n_iters += 1

        avg_loss = round(valid_loss/n_iters, 6)
        epoch_str = h.epoch_str(epoch)
        print('{}||POL Loss: {}'.format(epoch_str, avg_loss))
    

        self.model.train()
        
        return avg_loss

    def eval_on_env(self):
        eval_env = EvaluationEnvironment(self.ae, self.model, self.env_name, self.idx_list, self.config)
        avg_reward = eval_env.eval_mjc()
        
        return avg_reward

    def eval_on_ss(self):
        eval_env = EvaluationEnvironment(self.ae, self.model, self.env_name, self.idx_list, self.config)
        eval_env.eval_ss()


