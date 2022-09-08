import yaml
import random
import numpy as np
import torch
import os 
import pickle

def get_params(pth):
        with open(pth) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
            return config

def train_val_split(pth, frac):
    print("... read data")
    data_file = os.path.join(pth)

    f = open(data_file,'rb')
    data = pickle.load(f)
    n_samples = len(data['x'])

    idx = int((1-frac) * n_samples)
    # split data into training and validation set
    print("... splitting into train and val set")
    
    train_x = data['z'][idx:]
    train_y = data['y'][idx:]
    val_x = data['z'][:idx]
    val_y = data['y'][:idx]
    return train_x, train_y, val_x, val_y


def add_noise(states):
    
    noise_val = 0.002
    
    for i in range(len(states-1)):
        noise = (random.random()*2 - 1) * noise_val
        states[i] += noise
    
    return states

def backshift(x, dim):
    tmp = x[:-1]
    pad = torch.zeros(1, dim).cuda()
    res = torch.cat((pad, tmp), dim=0)
    return res

def loss_gaussian_nll(pred, label, size):
        mean = pred[:,:size]
        beta = pred[:,size:]
        # print("beta", beta)
        # print("mean", mean)
        assert not torch.isnan(pred).any()
        assert not torch.isnan(label).any()
        print(beta)
        tmp_1 = (-1 / 2) * torch.log(beta) 
        tmp_2 = (1 / 2) * beta * torch.square(mean - label)
        assert not torch.isnan(tmp_1).any()
        assert not torch.isnan(tmp_2).any()
        loss = tmp_1 + tmp_2 
        assert not torch.isnan(loss).any()
        mean_loss = torch.mean(loss)
        
        # print("label: ", label.shape)
        # print("mean: ", mean.shape)
        # print("var: ", var.shape)
        # print("loss: ", mean_loss)
        return mean_loss

def get_means(x, size):
    mu = x[:size]
    return mu

def get_vars(x, size):
    sig2 = x[size:]
    return sig2 