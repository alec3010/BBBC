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
    