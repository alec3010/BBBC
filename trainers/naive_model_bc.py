import torch
#from torch.utils.tensorboard import SummaryWriter


import os
import matplotlib as mpl
import matplotlib.pyplot as plt

import models.models as m

from trainers.behaviorcloner import BehaviorCloner





class NaiveModelBC(BehaviorCloner):
    def __init__(self, env_name, configs) -> None:
        super(NaiveModelBC, self).__init__(env_name=env_name, configs=configs)
        self.agent = m.model_factory(self.network_arch, self.obs_dim, self.acs_dim, configs=configs)
        self.optimizer = torch.optim.Adam(self.agent.parameters())# adam optimization
        self.criterion = torch.nn.MSELoss()# MSE loss
        self.process_data()

    def train_policy(self):
        
        print("... train model")

        # for traj in train:
        # form data loader for training
        train_dataset = torch.utils.data.TensorDataset(self.train_x, self.train_y)
        self.agent.train()

        
        for epoch in range(self.epochs):
            train_loss=0
            n_iters=0
            train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = self.batch_size, shuffle=self.shuffle)
            for batch_idx, (inputs,targets) in enumerate(train_loader):
                self.optimizer.zero_grad() # reset weights
                outputs = self.agent(inputs) # agent, pytorch
                loss = self.criterion(outputs,targets) # mse loss
                loss.backward() # backprop
                self.optimizer.step() # adam optim, gradient updates
                train_loss+=loss.item()
                n_iters+=1
            
            self.writer.add_scalar("Loss/train", (train_loss/n_iters), epoch)

            if (epoch+1)%self.eval_int == 0 and epoch != 0:
                self.eval_policy()

            print("average train batch loss in epoch " + str(epoch + 1) + ": " + str(train_loss / n_iters))
            self.result_dict['train_loss']['epoch'].append(epoch + 1)
            self.result_dict['train_loss']['value'].append(train_loss/n_iters)
            train_loss = 0
        
        reward = self.eval_on_env()

        print('Reward on Environment: %f' % reward )

    def eval_policy(self):
        val_dataset = torch.utils.data.TensorDataset(self.val_x, self.val_y)
        self.agent.eval()
        
            
        # form data loader for validation (currently predicts on whole valid set)
        valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size,shuffle=self.shuffle)
        valid_loss = 0
        n_iters = 0

        self.agent.eval()
        with torch.no_grad():
            for i, (valid_inputs,valid_targets) in enumerate(valid_loader):
                valid_inputs = valid_inputs.float()
                valid_outputs = self.agent(valid_inputs)
                valid_loss += self.criterion(valid_outputs,valid_targets).item()
                n_iters+=1

            print(f'valid set loss: {valid_loss/n_iters}')    
        
    def process_data(self):

        train, val = self.train_val_split()

        train_x = []
        train_y = []
        val_x = []
        val_y = []   

        for traj in train:
            for point in traj:
                obs = self.occlusion(point)
                tmp_x, tmp_y = obs, torch.from_numpy(point["acs"].astype(float)) 
                train_x.append(tmp_x.cuda())
                train_y.append(tmp_y.cuda())

            
        for traj in val:
            for point in traj:
                obs = self.occlusion(point)
                tmp_x, tmp_y = obs, torch.from_numpy(point["acs"].astype(float)) 
                val_x.append(tmp_x.cuda())
                val_y.append(tmp_y.cuda())


        self.train_x =torch.stack(train_x, 0).type(torch.cuda.FloatTensor)
        self.train_y =torch.stack(train_y, 0).type(torch.cuda.FloatTensor)
        self.val_x =torch.stack(val_x, 0).type(torch.cuda.FloatTensor)
        self.val_y =torch.stack(val_y, 0).type(torch.cuda.FloatTensor)

    