import os
import pickle
import numpy as np
import torch
import math
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR

from utils.pytorchtools import EarlyStopping
from utils import helpers as h
from utils.data_loader import DataLoader
from eval_env import EvaluationEnvironment
from models.GRUVAE import GRUVAE
from models.FF import FF
from trainers.trainer import Trainer

class VAETrainer(Trainer):

    def __init__(self, env_name, configs) -> None:
        super(VAETrainer,self).__init__(env_name=env_name, configs=configs)
        print("Creating VAETrainer Object")
        self.train_x, self.train_y, self.val_x, self.val_y  = h.train_val_split(self.db_path_belief, self.split, self.idx_list)
        if self.prev_acs:
            self.model = GRUVAE(self.obs_dim + self.acs_dim, self.acs_dim, self.acs_encoding_dim, self.obs_dim, self.belief_dim, self.hidden_dim, self.decoder_hidden, self.k, self.prev_acs).cuda()
        else:
            self.model = GRUVAE(self.obs_dim, self.acs_dim, self.acs_encoding_dim, self.obs_dim, self.belief_dim, self.hidden_dim, self.decoder_hidden, self.k, self.prev_acs).cuda()
        
        self.init_optimizer()
           
    def train(self):
        stopper = EarlyStopping(patience=50, verbose=False)
        print("... train vae model")
        self.model.train()
        k = self.k
        for epoch in range(self.epochs):
            
            n_iters = 0
            train_loss = 0.0
            
            for (x, y) in zip(self.train_x, self.train_y):

                future_acs = h.tm1_tpkm1(y, k)
                past_acs = h.tmkm1_tm1(y, k)
                    
                pred, mu, log_sigma = self.model(x[k+1:-k], y[k:-k-1], future_acs=future_acs, past_acs=past_acs) # vae, pytorch , mu_s, log_sigma_s 
                
                pred_labels = h.get_pred_labels(x, y, k)
                _, reg_loss = self.reg_loss(pred, pred_labels)
                kld = h.kl_div(mu, log_sigma)
                loss = reg_loss + kld * self.loss_weights['kld']
                self.opt.zero_grad() # reset weights
                loss.backward() # backprop
                self.opt.step() # adam optim, gradient update
                train_loss += loss.item()
                n_iters+=1
            if (epoch+1)%50 == 0 and epoch != 0: 
                self.scheduler.step()
            
            self.writer.add_scalar("LossVAE/TRAIN", (train_loss/n_iters), epoch + 1)   
            # print("Training loss in epoch " + str(epoch + 1) + ": " + str(train_loss / n_iters))
            # self.result_dict['vae_train_loss']['epoch'].append(epoch + 1)
            # self.result_dict['vae_train_loss']['value'].append(train_loss/n_iters)
            
            if (epoch+1)%self.eval_int == 0 and epoch != 0:
                
                val_loss = self.eval(epoch)
                   
                stopper(val_loss, self.model)
                if stopper.early_stop:
                    print("Early stop")
                    break

        self.model.save(self.vae_state_dict)

    def eval(self, epoch):     
        valid_loss, valid_loss_rec, valid_loss_kld = 0.0, 0.0, 0.0
        valid_loss_rec, valid_loss_obs_1fwd, valid_loss_obs_1bwd, valid_loss_obs_kfwd, valid_loss_obs_kbwd, valid_loss_acs_1fwd, valid_loss_acs_kfwd = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        self.model.eval()
        k = self.k
        n_iters = 0
        for (x, y) in zip(self.val_x, self.val_y):
            future_acs = h.tm1_tpkm1(y, k)
            past_acs = h.tmkm1_tm1(y, k)
            pred, mu, log_sigma, _ = self.model(x[k+1:-k], y[k:-k-1], future_acs=future_acs, past_acs=past_acs)
            pred_labels = h.get_pred_labels(x, y, k)
            reg_loss_dict, reg_loss = self.reg_loss(pred, pred_labels)
            kld = h.kl_div(mu, log_sigma)
            loss = reg_loss + kld * self.loss_weights['kld']
            # valid_loss_rec += reg_loss_dict['reconstruction'].item()
            valid_loss_obs_1fwd += reg_loss_dict['obs_one_fwd'].item()
            valid_loss_obs_1bwd += reg_loss_dict['obs_one_bwd'].item()
            valid_loss_obs_kfwd += reg_loss_dict['obs_k_fwd'].item()
            valid_loss_obs_kbwd += reg_loss_dict['obs_k_bwd'].item()
            # valid_loss_acs_1fwd += reg_loss_dict['acs_1_fwd'].item()
            # valid_loss_acs_kfwd += reg_loss_dict['acs_k_fwd'].item()
            valid_loss_kld += kld.item()
            valid_loss += loss.item()
            n_iters += 1

        avg_loss = round(valid_loss/n_iters, 6)
        avg_rec_loss = round(valid_loss_rec/n_iters, 6)
        avg_obs_1fwd_loss = round(valid_loss_obs_1fwd/n_iters, 6)
        avg_obs_1bwd_loss = round(valid_loss_obs_1bwd/n_iters, 6)
        avg_obs_kfwd_loss = round(valid_loss_obs_kfwd/n_iters, 6)
        avg_obs_kbwd_loss = round(valid_loss_obs_kbwd/n_iters, 6)
        # avg_acs_1fwd_loss = round(valid_loss_acs_1fwd/n_iters, 6)
        # avg_acs_kfwd_loss = round(valid_loss_acs_kfwd/n_iters, 6)
        avg_kld_loss = round(valid_loss_kld/n_iters, 6)
        epoch_str = h.epoch_str(epoch)
        print('VAE {}|| TOT: {} || OBS || REC: N/A | 1FWD: {} | 1BWD: {} | KFWD: {} | KBWD: {} | ACS || 1FWD: N/A | KFWD: N/A KLD: {} '\
            .format(epoch_str, 
                    avg_loss, 
                    avg_obs_1fwd_loss, 
                    avg_obs_1bwd_loss, 
                    avg_obs_kfwd_loss, 
                    avg_obs_kbwd_loss,
                    avg_kld_loss))
        
        self.writer.add_scalar("LossVAE/VAL", (avg_loss), epoch + 1)
        # self.writer.add_scalar("LossVAE/REC", (avg_rec_loss), epoch + 1)
        self.writer.add_scalar("LossVAE/OBS/1FWD", (avg_obs_1fwd_loss), epoch + 1)
        self.writer.add_scalar("LossVAE/OBS/1BWD", (avg_obs_1bwd_loss), epoch + 1)
        self.writer.add_scalar("LossVAE/OBS/KFWD", (avg_obs_kfwd_loss), epoch + 1)
        self.writer.add_scalar("LossVAE/OBS/KBWD", (avg_obs_kbwd_loss), epoch + 1)
        # self.writer.add_scalar("LossVAE/ACS/1FWD", (avg_acs_1fwd_loss), epoch + 1)
        # self.writer.add_scalar("LossVAE/ACS/KFWD", (avg_acs_kfwd_loss), epoch + 1)
        self.writer.add_scalar("LossVAE/KLD", (avg_kld_loss), epoch + 1)
        # self.result_dict['val_loss']['value'].append(avg_loss)

        self.model.train()

        return avg_loss

    def reg_loss(self, pred, labels):
        losses = {}
        loss = 0
        for key in pred:
            
            assert not torch.isnan(pred[key]).any(), "NaN at key '{}'".format(key)
            assert torch.is_tensor(labels[key])
            assert not torch.isnan(labels[key]).any()
            if not pred[key].size() == labels[key].size():
                print('prediction size:', pred[key].size())
                print('label size:', labels[key].size())
            assert pred[key].size() == labels[key].size(), "Sizes did not Match at key '{}'".format(key)

            losses[key] = self.mse(pred[key], labels[key])
            loss += losses[key]*self.loss_weights[key]
        return losses, loss
