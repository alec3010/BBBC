import os

import torch
import torch.nn as nn
import torch.nn.functional as f
import yaml

from utils import helpers as h
import utils.encoder_decoder as ed

from models.FF import FF 
from models.RNNVAE import RNNVAE


def model_factory(type, obs_dim, acs_dim, configs):
    if type == "FF":
        return FF(obs_dim=obs_dim, acs_dim=acs_dim, config=configs).cuda()

    elif type == "RNNVAE":
        model = RNNVAE(obs_dim=obs_dim, acs_dim=acs_dim, config=configs).cuda()
        return model



            
        

