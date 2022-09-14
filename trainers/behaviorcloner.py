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

class BehaviorCloner():

    def __init__(self, env_name, configs) -> None:
        print("Creating Behavior cloner Object")
        
        self.config = configs
        self.get_params()
        self.env_name = env_name
        self.load_dataset_idx()
        print("Configs are: ","\n" , self.config)
        self.vae = GRUVAE(self.obs_dim, self.belief_dim, self.hidden_dim, self.decoder_hidden).cuda()
        self.ff_policy = FF(self.acs_dim, self.belief_dim, self.policy_hidden).cuda()
        self.init_optimizers()
        
        self.train_x, self.train_y, self.val_x, self.val_y  = h.train_val_split(self.db_path, self.split)
        # self.train_x, self.train_y, self.val_x, self.val_y  = h.generate_sin_dataset(50000, self.split)

        self.writer = SummaryWriter(log_dir = "./tensorboard")
        self.model_dir = "./models"
        
        self.reset_results()

    def train_vae(self):
        stopper = EarlyStopping(patience=100, verbose=False)
        print("... train vae model")
        self.vae.train()
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
                
                pred, mu_s, sigma_s = self.vae(x[k:-k]) # vae, pytorch
                pred_labels = h.get_pred_labels(x, k)
            
                loss = self.loss(pred, pred_labels, mu_s,sigma_s)
                self.vae_opt.zero_grad() # reset weights
                
                loss.backward() # backprop
                self.vae_opt.step() # adam optim, gradient update
                
                train_loss+=loss.item()
                
                n_iters+=1
            if (epoch+1)%50 == 0 and epoch != 0: 
                self.scheduler.step()
            
            self.writer.add_scalar("LossVAE/TRAIN", (train_loss/n_iters), epoch + 1)   
            # print("Training loss in epoch " + str(epoch + 1) + ": " + str(train_loss / n_iters))
            # self.result_dict['vae_train_loss']['epoch'].append(epoch + 1)
            # self.result_dict['vae_train_loss']['value'].append(train_loss/n_iters)
            
            if (epoch+1)%self.eval_int == 0 and epoch != 0:
                
                val_loss = self.eval_vae(epoch)
                self.writer.add_scalar("LossVAE/VAL", (val_loss), epoch + 1)   
                stopper(val_loss, self.vae)
                if stopper.early_stop:
                    print("Early stop")
                    break

        # reward = self.eval_on_ss()
        # reward = self.eval_on_env()
        #print('Reward on Environment: %f' % reward )

    def eval_vae(self, epoch):
        
        loader = DataLoader(self.val_x,
                               self.val_y,
                               self.seq_length, 
                               self.idx_list,
                               self.network_arch)
        
        valid_loss, valid_loss_rec, valid_loss_kld, valid_var = 0.0, 0.0, 0.0, 0.0
        valid_loss_1fwd, valid_loss_1bwd, valid_loss_kfwd, valid_loss_kbwd = 0.0, 0.0, 0.0, 0.0
        self.vae.eval()
        k = self.k
        n_iters = 0
        for (x, y) in loader:
            pred, mu_s, sigma_s, hn = self.vae(x[k:-k]) # vae, pytorch

            pred_labels = h.get_pred_labels(x, k)
            rec_loss = self.rec_loss(pred, pred_labels)
            kld = h.kl_div(mu_s, sigma_s)
            
            loss = sum(rec_loss.values()) + self.loss_weights['latent']*kld
            valid_loss_rec += rec_loss['reconstruction'].item()
            valid_loss_1fwd += rec_loss['one_fwd'].item()
            valid_loss_1bwd += rec_loss['one_bwd'].item()
            valid_loss_kfwd += rec_loss['k_fwd'].item()
            valid_loss_kbwd += rec_loss['k_bwd'].item()
            valid_loss_kld += kld.item()
            valid_loss += loss.item()
            valid_var += torch.mean(sigma_s).item()
            n_iters += 1

        avg_loss = round(valid_loss/n_iters, 4)
        avg_rec_loss = round(valid_loss_rec/n_iters, 4)
        avg_1fwd_loss = round(valid_loss_1fwd/n_iters, 4)
        avg_1bwd_loss = round(valid_loss_1bwd/n_iters, 4)
        avg_kfwd_loss = round(valid_loss_kfwd/n_iters, 4)
        avg_kbwd_loss = round(valid_loss_kbwd/n_iters, 4)
        avg_kld_loss = round(valid_loss_kld/n_iters, 4)
        avg_var = round(valid_var/n_iters, 4)
        epoch_str = h.epoch_str(epoch)
        print('{}||VAE Losses TOT: {} | REC: {} | 1FWD: {} | 1BWD:{} | KFWD: {} | KBWD: {} | KLD: {} || Variance: {}'\
            .format(epoch_str, 
                    avg_loss, 
                    avg_rec_loss, 
                    avg_1fwd_loss, 
                    avg_1bwd_loss, 
                    avg_kfwd_loss, 
                    avg_kbwd_loss, 
                    avg_kld_loss, 
                    avg_var))
        self.writer.add_scalar("LossVAE/REC", (avg_rec_loss), epoch + 1)   
        self.writer.add_scalar("LossVAE/1FWD", (avg_1fwd_loss), epoch + 1)
        self.writer.add_scalar("LossVAE/1BWD", (avg_1bwd_loss), epoch + 1)
        self.writer.add_scalar("LossVAE/KFWD", (avg_kfwd_loss), epoch + 1)
        self.writer.add_scalar("LossVAE/KBWD", (avg_kbwd_loss), epoch + 1)
        self.writer.add_scalar("LossVAE/KLD", (avg_kld_loss), epoch + 1)
        
        self.result_dict['val_loss']['value'].append(avg_loss)
            #print(f'valid set accuracy: {valid_acc}')

        self.vae.train()
        return avg_loss

    def train_policy(self):
        stopper = EarlyStopping(patience=100, verbose=False)
        print("... train policy model")
        self.ff_policy.train()
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
                
                pred, mu_s, sigma_s = self.vae(x[k:-k]) # vae, pytorch
                acs = self.ff_policy(mu_s.detach())
                loss = self.mse(acs, y[k:-k])
                
                self.pol_opt.zero_grad() # reset weights
                loss.backward() # backprop
                self.pol_opt.step() # adam optim, gradient update
                train_loss+=loss.item()
                n_iters+=1
            if (epoch+1)%50 == 0 and epoch != 0: 
                self.scheduler.step()
            
            self.writer.add_scalar("LossPOL/TRAIN", (train_loss/n_iters), epoch + 1)   
            # print("Training loss in epoch " + str(epoch + 1) + ": " + str(train_loss / n_iters)
            # self.result_dict['train_loss']['value'].append(train_loss/n_iters)
            
            if (epoch+1)%self.eval_int == 0 and epoch != 0:
                
                val_loss = self.eval_policy(epoch)
                self.writer.add_scalar("LossPOL/VAL", (val_loss), epoch + 1)   
                stopper(val_loss, self.vae)
                if stopper.early_stop:
                    print("Early stop")
                    break
        pass

    def eval_policy(self, epoch):
        loader = DataLoader(self.val_x,
                               self.val_y,
                               self.seq_length, 
                               self.idx_list,
                               self.network_arch)
        
        valid_loss = 0.0
        self.vae.eval()
        self.ff_policy.eval()
        k = self.k
        n_iters = 0
        for (x, y) in loader:
            
            pred, mu_s, sigma_s, _ = self.vae(x[k:-k]) # vae, pytorch
            acs = self.ff_policy(mu_s.detach())
            loss = self.mse(acs, y[k:-k])
            valid_loss += loss.item()
            n_iters += 1

        avg_loss = round(valid_loss/n_iters, 4)
        epoch_str = h.epoch_str(epoch)
        print('{}||POL Loss: {}'.format(epoch_str, avg_loss))
    

        self.ff_policy.train()
        self.vae.train()
        return avg_loss



    def train(self):
        self.train_vae()
        self.train_policy()


        

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
            assert pred[key].size() == labels[key].size(), "Sizes did not Match at key '{}'".format(key)
            losses[key] = self.mse(pred[key], labels[key])*self.loss_weights['decoder'][key]
        return losses


    def init_optimizers(self):
        self.vae_opt = torch.optim.Adam(self.vae.parameters(),lr=self.lr)# adam optimization
        self.pol_opt = torch.optim.Adam(self.ff_policy.parameters(), lr=self.lr)
        self.scheduler = ExponentialLR(optimizer=self.vae_opt, gamma=self.gamma, verbose=False)
        self.mse = torch.nn.MSELoss()# MSE loss

    def save_model(self):
        torch.save(self.vae.state_dict(), os.path.join(model_dir,"InvertedPendulum.pkl"))

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
        self.k = self.config['k']
        self.loss_weights = self.config['loss_weights']
        self.hidden_dim = self.config['hidden_dim']
        self.decoder_hidden = self.config['decoder_hidden']
        self.belief_dim = self.config['belief_dim']
        self.policy_hidden = self.config['policy_hidden']

    def load_dataset_idx(self):
        dataset_idx = h.get_params("configs/dataset_index.yaml")
        entry = dataset_idx[self.env_name]
        self.db_path = entry['db']
        self.obs_dim = len(entry['obs_dim'][self.process_model])
        self.acs_dim = entry['acs_dim']
        self.idx_list = entry['obs_dim'][self.process_model]

    def eval_on_env(self):
        eval_env = EvaluationEnvironment(self.vae, self.env_name, self.idx_list, self.config)
        avg_reward = eval_env.eval_mjc()
        
        return avg_reward

    def eval_on_ss(self):
        eval_env = EvaluationEnvironment(self.vae, self.env_name, self.idx_list, self.config)
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

    

    

    

    
        
        

    
        

    



