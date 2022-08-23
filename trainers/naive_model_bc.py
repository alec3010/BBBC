import torch
#from torch.utils.tensorboard import SummaryWriter


import os
import matplotlib as mpl
import matplotlib.pyplot as plt

import models as m

from trainers.behaviorcloner import BehaviorCloner




class NaiveModelBC(BehaviorCloner):
    def __init__(self, env_name) -> None:
        super(NaiveModelBC, self).__init__(env_name=env_name)
        self.agent = m.model_factory(self.network_arch, self.obs_dim, self.acs_dim)
        self.optimizer = torch.optim.Adam(self.agent.parameters())# adam optimization
        self.criterion = torch.nn.MSELoss()# MSE loss
        self.traj_nr = 0
        self.batch_size = 32
        self.shuffle = True
        
        self.process_data()

    def train_policy(self):
        
        print("... train model")

        # for traj in train:
        # form data loader for training
        train_dataset = torch.utils.data.TensorDataset(self.train_x, self.train_y)
        self.agent.train()

        n_iters = 0
        train_loss, train_cor = 0,0
        for i in range(self.episodes):
            train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = self.batch_size, shuffle=self.shuffle)
            for batch_idx, (inputs,targets) in enumerate(train_loader):
                self.optimizer.zero_grad() # reset weights
                outputs = self.agent(inputs) # agent, pytorch
                loss = self.criterion(outputs,targets) # mse loss
                loss.backward() # backprop
                self.optimizer.step() # adam optim, gradient updates
                train_loss+=loss.item()
                n_iters+=1
                self.writer.add_scalar("Loss/train", loss.item(), n_iters)
                
    
            print(f'average train batch loss: {(train_loss / n_iters)}')

    def eval():

        print(f'average train batch accuracy: {(train_cor / (batch_size*n_iters))}')
            
        # form data loader for validation (currently predicts on whole valid set)
        for demo in val:
            valid_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(demo["obs"]),torch.FloatTensor(demo["acs"]))
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=len(demo["obs"]),shuffle=False)
            valid_loss, valid_acc = 0,0
            self.agent.eval()
            with torch.no_grad():
                for i, (valid_inputs,valid_targets) in enumerate(valid_loader):
                    valid_inputs = valid_inputs.unsqueeze(1).float()
                    valid_outputs = self.agent(valid_inputs)
                    valid_loss += self.criterion(valid_outputs,valid_targets).item()
                    """ accuracy
                    _, valid_predicted = torch.max(torch.abs(valid_outputs),1) 
                    _, valid_targetsbinary = torch.max(torch.abs(valid_targets),1)
                    valid_correct = (valid_predicted==valid_targetsbinary).sum().item()
                    valid_acc+=(valid_correct/valid_targets.shape[0])
                    """
                print(f'valid set loss: {valid_loss}')
                #print(f'valid set accuracy: {valid_acc}')

        
        
        print("Model saved in file: %s" % model_dir)
        
    def process_data(self):

        train, val = self.train_val_split()

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


        self.train_x =torch.stack(train_x, 0).type(torch.cuda.FloatTensor)
        self.train_y =torch.stack(train_y, 0).type(torch.cuda.FloatTensor)
        print(self.train_x.size())
        print(self.train_y.size())
        self.val_x =torch.stack(val_x, 0).type(torch.cuda.FloatTensor)
        self.val_y =torch.stack(val_y, 0).type(torch.cuda.FloatTensor)

    