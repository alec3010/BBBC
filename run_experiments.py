
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from pomdpbc import POMDPBC

import utils.helpers as h

if __name__ == "__main__":

    # read data    
    train, val = h.train_val_split_pomdp()
    bc = POMDPBC(lr=0.001, belief_dim=64, action_dim=1, obs_dim=2, actor_hidden_dim=64)

    bc.train_model(train, val, batch_size=8)
