import yaml
import random
import numpy as np
import torch
import os 
import pickle
import math

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



def get_pred_labels(x, k):
    labels = {}
    labels['reconstruction'] = x[k:-k]
    labels['one_fwd'] = x[k+1:-k+1,:]
    labels['one_bwd'] = x[k-1:-k-1,:]
    labels['k_fwd'] = x[2*k:,:]
    labels['k_bwd'] = x[:-2*k,:]
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


def kl_div(mu, sigma):
    if not len(sigma[sigma==0])==0:
        print(len(sigma[sigma==0]))
    assert len(sigma[sigma==0])==0
    dot_product = torch.sum(mu*mu, dim=-1)
    trace = torch.sum(sigma, dim=-1)
    k = torch.Tensor(mu.size(0)).cuda()
    k.fill_(mu.size(1))
    cov  = torch.diag_embed(sigma)
    log = torch.log(torch.det(cov))
    

    assert len(dot_product[dot_product<0])==0
    assert len(trace[trace<0])==0
    assert len(k[k<0])==0

    kl_vec = (1/2) * (dot_product + trace - k - log)
    result = torch.sum(kl_vec)/mu.size(0)
    
    return result

def epoch_str(epoch):
    epoch_str = ""
    if math.log10(epoch+1) < 2:
        epoch_str = str(epoch+1) + "  "
    elif math.log10(epoch+1) < 3:
        epoch_str = str(epoch+1) + " "
    elif math.log10(epoch+1) < 4:
        epoch_str = str(epoch+1)
    return epoch_str