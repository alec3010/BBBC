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

def bwd_shift(x,k):
    x = x[:-2*k,:]
    return x

def fwd_shift(x,k):
    x = x[2*k:,:]
    return x

def one_bwd_shift(x,k):
    x = x[k-1:-k-1,:]
    return x

def one_fwd_shift(x,k):
    x = x[k+1:-k+1,:]
    return x

def get_pred_labels(x, k):
    labels = {}
    labels['one_fwd'] = one_fwd_shift(x,k)
    labels['one_bwd'] = one_bwd_shift(x,k)
    labels['k_fwd'] = fwd_shift(x,k)
    labels['k_bwd'] = bwd_shift(x,k)
    return labels


def loss_gaussian_nll(mean, beta, label):
        assert not torch.isnan(mean).any()
        assert not torch.isnan(beta).any()
        assert not torch.isnan(label).any()   
        loss = (-1 / 2) * torch.log(beta) + (1 / 2) * beta * torch.square(mean - label)
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