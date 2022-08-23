import os
import pickle
import torch

def process_data_pomdp(train, val):
    """
    This method reads the states and actions recorded from the RL agent 
    and splits it into training/ validation set for behavior cloning in chronological order.
    """

    data = train_val_split()

    for traj in train:
        for point in traj:
            tmp_x, tmp_y = torch.from_numpy(point["obs"].astype(float)), torch.from_numpy(point["acs"].astype(float)) 
            point["obs"], point["acs"] = tmp_x.cuda(), tmp_y.cuda()
        
    for traj in val:
        for point in traj:
            tmp_x, tmp_y = torch.from_numpy(point["obs"].astype(float)), torch.from_numpy(point["acs"].astype(float)) 
            point["obs"], point["acs"] = tmp_x.cuda(), tmp_y.cuda()

    return train, val



def process_data_mdp(train, val):
    """
    This method reads the states and actions recorded from the RL agent 
    and splits it into training/ validation set for behavior cloning in minibatches.
    """

   
    train_x = []
    train_y = []
    val_x = []
    val_y = []   

    for traj in train:
        for point in traj:
            tmp_x, tmp_y = torch.from_numpy(point["obs"].astype(float)), torch.from_numpy(point["acs"].astype(float)) 
            train_x.append(tmp_x.cuda())
            train_y.append(tmp_y.cuda())
        
    for traj in val:
        for point in traj:
            tmp_x, tmp_y = torch.from_numpy(point["obs"].astype(float)), torch.from_numpy(point["acs"].astype(float)) 
            val_x.append(tmp_x.cuda())
            val_y.append(tmp_y.cuda())


    train_x =torch.cat(train_obs, 0).type(torch.cuda.FloatTensor)
    train_y =torch.cat(train_acs, 0).type(torch.cuda.FloatTensor)
    val_x =torch.cat(val_obs, 0).type(torch.cuda.FloatTensor)
    val_y =torch.cat(val_acs, 0).type(torch.cuda.FloatTensor)
    
    return train_x, train_y, val_x, val_y