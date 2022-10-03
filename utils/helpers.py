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

def train_val_split(pth, frac, idx_list):
    print("... read data")
    data_file = os.path.join(pth)

    f = open(data_file,'rb')
    data = pickle.load(f)
    n_samples = len(data['z'])

    idx = int((1-frac) * n_samples)
    # split data into training and validation set
    print("... splitting into train and val set")
    for traj in data['z']:
        for idx, p in enumerate(traj):
            tmp = []
            for j in idx_list:

                tmp.append(p[j])
            traj[idx] = np.array(tmp)
            
    train_x_ = data['z'][idx:]
    train_y_ = data['y'][idx:]

    val_x_ = data['z'][:idx]
    val_y_ = data['y'][:idx]
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    for x in train_x_:
        tmp_x = np.array(x)
        train_x.append(torch.Tensor(tmp_x).squeeze().cuda())
    for y in train_y_:
        tmp_y = np.array(y)
        train_y.append(torch.Tensor(tmp_y).cuda())
    for x in val_x_:
        tmp_x = np.array(x)
        val_x.append(torch.Tensor(tmp_x).squeeze().cuda())
    for y in val_y_:
        tmp_y = np.array(y)
        val_y.append(torch.Tensor(tmp_y).cuda())
    

    
    return train_x, train_y, val_x, val_y




def add_noise(states):
    
    noise_val = 0.02
    
    for i in range(len(states-1)):
        noise = (random.random()*2 - 1) * noise_val
        states[i] += noise
    
    return states



def get_pred_labels(x, y, k):
    labels = {}
    labels['reconstruction'] = x[k:-k]
    labels['one_fwd'] = x[k+1:-k+1,:]
    labels['one_bwd'] = x[k-1:-k-1,:]
    labels['k_fwd'] = x[2*k:,:]
    labels['k_bwd'] = x[:-2*k,:]
    labels['acs'] = inject_gaussian_noise(y[k:-k], 10)
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

def kl_div(mu, log_sigma):
    result = -0.5 * torch.sum(1 + log_sigma - mu**2 - torch.exp(log_sigma))
    return result

def inject_gaussian_noise(tensor, sigma):
    mean_tensor = torch.zeros_like(tensor)
    sigma_tensor = torch.ones_like(tensor)*sigma
    noise = torch.normal(mean_tensor, sigma_tensor) 
    return tensor + noise


def epoch_str(epoch):
    epoch_str = ""
    if math.log10(epoch+1) < 2:
        epoch_str = str(epoch+1) + "  "
    elif math.log10(epoch+1) < 3:
        epoch_str = str(epoch+1) + " "
    elif math.log10(epoch+1) < 4:
        epoch_str = str(epoch+1)
    return epoch_str

def generate_sin_dataset(length, frac):
    t = 0
    x = []
    z = []
    t = np.expand_dims(np.arange(0, 4*math.pi, 4*math.pi/length, dtype=float), axis=1)
    
    x = np.sin(t) 
    # add gaussian zero-mean noise
    noise = np.expand_dims(np.random.normal(0, 0.03, size=t.shape), axis=1)
    
    z = x + noise 


    idx = int((1-frac) * len(x))
    # split data into training and validation set
    print("... splitting into train and val set")
    
    train_x = z[idx:]
    train_y = x[idx:]
    val_x = z[:idx]
    val_y = x[:idx]
    return train_x, train_y, val_x, val_y