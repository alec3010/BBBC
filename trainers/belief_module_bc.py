import os
import numpy as np
import torch
#from torch.utils.tensorboard import SummaryWriter

import models.models as m
from trainers.behaviorcloner import BehaviorCloner

import utils.helpers as h


class BeliefModuleBC(BehaviorCloner):
    def __init__(self, env_name, configs) -> None:
        super(BeliefModuleBC, self).__init__(env_name=env_name, configs=configs)
        self.agent = m.model_factory(configs['network_arch'], obs_dim=self.obs_dim, acs_dim=self.acs_dim, configs=configs)
        self.optimizer = torch.optim.Adam(self.agent.parameters(),lr=self.lr)# adam optimization
        self.criterion = torch.nn.MSELoss()# MSE loss
        #self.writer = SummaryWriter(log_dir = "./tensorboard")
        self.traj_nr = 0
        
    
        self.process_data(configs['traj_length'])

        

    def train_policy(self):
        print("... train model")
        self.agent.train()
        steps = 0
        for epoch in range(self.epochs):
            n_iters = 0
            train_loss = 0.0
            self.agent.reset_memory()
            for (traj_x, traj_y) in zip(self.train_x, self.train_y):
                # for point in traj:
                #     outputs, belief = self.agent(self.curr_memory) # agent, pytorch
                    
                #     loss = self.criterion(outputs, point['acs']) # mse loss
                    
                #     loss.backward(retain_graph=True) # backprop
                #     self.optimizer.step() # adam optim, gradient updates

                #     train_loss+=loss.item()
                #     n_iters+=1
                # self.writer.add_scalar("Loss/train", loss.item(), n_iters)
                
                outputs = self.agent(traj_x) # agent, pytorch
                forces = torch.mean(outputs)
                self.writer.add_scalar("Force", (forces.item()), steps)   
                loss = self.criterion(outputs, traj_y) # mse loss
                self.optimizer.zero_grad() # reset weights
                loss.backward() # backprop
                self.optimizer.step() # adam optim, gradient update
                train_loss+=loss.item()
                n_iters+=1
                steps += 1
            
            self.writer.add_scalar("Loss/train", (train_loss/n_iters), epoch + 1)   
            #print("average train trajectory loss in epoch " + str(epoch + 1) + ": " + str(train_loss / n_iters))
            self.result_dict['train_loss']['epoch'].append(epoch + 1)
            self.result_dict['train_loss']['value'].append(train_loss/n_iters)
            train_loss = 0
            
            if (epoch+1)%self.eval_int == 0 and epoch != 0:
                val_loss = self.eval_policy(epoch)
                self.writer.add_scalar("Loss/val", (val_loss), epoch + 1)   
        
        reward = self.eval_on_ss()

        #print('Reward on Environment: %f' % reward )
                

    def eval_policy(self, epoch):
        
        valid_loss= 0
        self.agent.eval()
        with torch.no_grad():
            for (traj_x, traj_y) in zip(self.val_x, self.val_y):
                
                outputs = self.agent(traj_x) # agent, pytorch
                valid_loss += self.criterion(outputs, traj_y).item() # mse loss

        avg_loss = valid_loss/len(self.val_x)
        print(f'valid set loss: {avg_loss}')
        self.result_dict['val_loss']['epoch'].append(epoch + 1)
        self.result_dict['val_loss']['value'].append(avg_loss)
            #print(f'valid set accuracy: {valid_acc}')

        self.agent.train()
        return avg_loss

    def process_data(self, length):

        
        train, val = self.train_val_split()
        self.train_x = []
        self.train_y = []
        
        self.val_x = []
        self.val_y = []

        for _ in train:
            traj_x = []
            traj_y = []
            for i in range(length):
                tmp_y = torch.from_numpy(_[i]["acs"].astype(float)) 

                _[i]["obs"], _[i]["acs"] = self.occlusion(_[i]), tmp_y.cuda()
                traj_x.append(_[i]['obs'])
                traj_y.append(_[i]['acs'])
            traj_x = torch.stack(traj_x, dim=0)
            traj_y = torch.stack(traj_y, dim=0)
            
            
            self.train_x.append(traj_x)
            self.train_y.append(traj_y)
            
        for _ in val:
            traj_x = []
            traj_y = []
            for i in range(length):
                tmp_y = torch.from_numpy(_[i]["acs"].astype(float)) 


                _[i]["obs"], _[i]["acs"] = self.occlusion(_[i]), tmp_y.cuda()
                traj_x.append(_[i]['obs'])
                traj_y.append(_[i]['acs'])
            traj_x = torch.stack(traj_x, dim=0)
            traj_y = torch.stack(traj_y, dim=0)
            
            self.val_x.append(traj_x)
            self.val_y.append(traj_y)
        
