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

        self.model = GRUVAE(self.obs_dim, self.acs_dim, self.belief_dim, self.hidden_dim, self.decoder_hidden).cuda()
        self.init_optimizer()
           
    def train(self):
        stopper = EarlyStopping(patience=10, verbose=False)
        print("... train vae model")
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
                pred, mu_s, sigma_s= self.model(x[k:-k]) # vae, pytorch , mu_s, sigma_s 
                pred_labels = h.get_pred_labels(x, y, k)
                loss = self.loss(pred, pred_labels, mu_s, sigma_s)
                self.opt.zero_grad() # reset weights
                loss.backward() # backprop
                self.opt.step() # adam optim, gradient update
                train_loss+=loss.item()
                n_iters+=1
            if (epoch+1)%50 == 0 and epoch != 0: 
                self.scheduler.step()
            
            self.writer.add_scalar("LossVAE/TRAIN", (train_loss/n_iters), epoch + 1)   
            # print("Training loss in epoch " + str(epoch + 1) + ": " + str(train_loss / n_iters))
            # self.result_dict['vae_train_loss']['epoch'].append(epoch + 1)
            # self.result_dict['vae_train_loss']['value'].append(train_loss/n_iters)
            
            if (epoch+1)%self.eval_int == 0 and epoch != 0:
                
                val_loss = self.eval(epoch)
                self.writer.add_scalar("LossVAE/VAL", (val_loss), epoch + 1)   
                stopper(val_loss, self.model)
                if stopper.early_stop:
                    print("Early stop")
                    break

        self.model.save(self.vae_state_dict)

    def eval(self, epoch):
        loader = DataLoader(self.val_x,
                               self.val_y,
                               self.seq_length, 
                               self.idx_list,
                               self.network_arch)
        
        valid_loss, valid_loss_rec, valid_loss_kld, valid_var = 0.0, 0.0, 0.0, 0.0
        valid_loss_1fwd, valid_loss_1bwd, valid_loss_kfwd, valid_loss_kbwd, valid_loss_acs = 0.0, 0.0, 0.0, 0.0, 0.0
        self.model.eval()
        k = self.k
        n_iters = 0
        for (x, y) in loader:
            pred, mu_s, sigma_s, _ = self.model(x[k:-k]) # vae, pytorch

            pred_labels = h.get_pred_labels(x, y, k)
            rec_loss = self.rec_loss(pred, pred_labels)
            kld = h.kl_div(mu_s, sigma_s)
            
            loss = sum(rec_loss.values()) + self.loss_weights['latent']*kld
            valid_loss_rec += rec_loss['reconstruction'].item()
            valid_loss_1fwd += rec_loss['one_fwd'].item()
            valid_loss_1bwd += rec_loss['one_bwd'].item()
            valid_loss_kfwd += rec_loss['k_fwd'].item()
            valid_loss_kbwd += rec_loss['k_bwd'].item()
            valid_loss_acs += rec_loss['acs'].item()
            valid_loss_kld += kld.item()
            valid_loss += loss.item()
            # valid_mean = torch.mean(mu_s.detach(), 0).cpu().numpy()
            # valid_var = torch.mean(sigma_s.detach(), 0).cpu().numpy()
            n_iters += 1

        avg_loss = round(valid_loss/n_iters, 4)
        avg_rec_loss = round(valid_loss_rec/n_iters, 4)
        avg_1fwd_loss = round(valid_loss_1fwd/n_iters, 4)
        avg_1bwd_loss = round(valid_loss_1bwd/n_iters, 4)
        avg_kfwd_loss = round(valid_loss_kfwd/n_iters, 4)
        avg_kbwd_loss = round(valid_loss_kbwd/n_iters, 4)
        avg_acs_loss = round(valid_loss_acs/n_iters, 4)
        avg_kld_loss = round(valid_loss_kld/n_iters, 4)
        avg_var = round(valid_var/n_iters, 4)
        epoch_str = h.epoch_str(epoch)
        print('{}||VAE Losses TOT: {} | REC: {} | 1FWD: {} | 1BWD: {} | KFWD: {} | KBWD: {} | ACS: {} | KLD: {}'\
            .format(epoch_str, 
                    avg_loss, 
                    avg_rec_loss, 
                    avg_1fwd_loss, 
                    avg_1bwd_loss, 
                    avg_kfwd_loss, 
                    avg_kbwd_loss,
                    avg_acs_loss,
                    avg_kld_loss))
        self.writer.add_scalar("LossVAE/REC", (avg_rec_loss), epoch + 1)   
        self.writer.add_scalar("LossVAE/1FWD", (avg_1fwd_loss), epoch + 1)
        self.writer.add_scalar("LossVAE/1BWD", (avg_1bwd_loss), epoch + 1)
        self.writer.add_scalar("LossVAE/KFWD", (avg_kfwd_loss), epoch + 1)
        self.writer.add_scalar("LossVAE/KBWD", (avg_kbwd_loss), epoch + 1)
        self.writer.add_scalar("LossVAE/ACS", (avg_acs_loss), epoch + 1)
        # self.writer.add_scalar("LossVAE/KLD", (avg_kld_loss), epoch + 1)
        
        self.result_dict['val_loss']['value'].append(avg_loss)
            #print(f'valid set accuracy: {valid_acc}')

        self.model.train()
        return avg_loss

    def loss(self, pred, rec_labels, mu, sigma):
        loss = 0
        loss += sum(self.rec_loss(pred, rec_labels).values())
        kld = h.kl_div(mu, sigma)
        loss += self.loss_weights['latent'] * kld
        return loss

    def rec_loss(self, pred, labels):
        losses = {}
        for key in pred:
            
            assert not torch.isnan(pred[key]).any()
            assert torch.is_tensor(labels[key])
            assert not torch.isnan(labels[key]).any()
            if not pred[key].size() == labels[key].size():
                print('prediction size:', pred[key].size())
                print('label size:', labels[key].size())
            assert pred[key].size() == labels[key].size(), "Sizes did not Match at key '{}'".format(key)

            losses[key] = self.mse(pred[key], labels[key])*self.loss_weights['decoder'][key]
        return losses
