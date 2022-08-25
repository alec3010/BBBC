import os
import numpy as np
import torch
#from torch.utils.tensorboard import SummaryWriter

import models as m
from trainers.behaviorcloner import BehaviorCloner

import utils.helpers as h


class BeliefModuleBC(BehaviorCloner):
    def __init__(self, env_name, configs) -> None:
        super(BeliefModuleBC, self).__init__(env_name=env_name, configs=configs)
        self.agent = m.model_factory("belief", obs_dim=self.obs_dim, acs_dim=self.acs_dim, configs=configs)
        self.init_optimizer()
        #self.writer = SummaryWriter(log_dir = "./tensorboard")
        self.traj_nr = 0
        
        self.reset_memory()
        self.process_data(configs['traj_length'])

        

    def train_policy(self):
        print("... train model")
        self.agent.train()
        for epoch in range(self.epochs):
            n_iters = 0
            train_loss = 0
            self.reset_memory()
            
            for traj in self.train:
                self.reset_memory()
                for point in traj:
                    self.curr_memory['curr_ob'] = point['obs']
                    self.optimizer.zero_grad() # reset weights
                    outputs, belief = self.agent(self.curr_memory) # agent, pytorch
                    self.curr_memory['prev_belief'] = belief.detach()
                    self.curr_memory['prev_ac'] = outputs.detach()
                    
                    #self.curr_memory['prev_obs'] = self.curr_memory['curr_ob']
                    
                    loss = self.criterion(outputs, point['acs']) # mse loss
                    
                    loss.backward(retain_graph=True) # backprop
                    self.optimizer.step() # adam optim, gradient updates

                    train_loss+=loss.item()
                    n_iters+=1
                    # self.writer.add_scalar("Loss/train", loss.item(), n_iters)
            
            self.writer.add_scalar("Loss/train", (train_loss/n_iters), epoch + 1)   
            print("average train trajectory loss in epoch " + str(epoch + 1) + ": " + str(train_loss / n_iters))
            self.result_dict['train_loss']['epoch'].append(epoch + 1)
            self.result_dict['train_loss']['value'].append(train_loss/n_iters)
            train_loss = 0
            
            if (epoch+1)%self.eval_int == 0 and epoch != 0:
                self.eval_policy(epoch)
                
        self.reset_memory()
        reward = self.eval_on_env()
        self.result_dict['reward'] = reward

        print('Reward on Environment: %f' % reward )
        #torch.save(self.agent.state_dict(), os.path.join(model_dir,"InvertedPendulum.pkl"))
        #print("Model saved in file: %s" % model_dir)

    def eval_policy(self, epoch):
        self.reset_memory()
        valid_loss= 0
        self.agent.eval()
        with torch.no_grad():
            for traj in self.val:
                for point in traj:
                    self.curr_memory['curr_ob'] = point['obs']
                    targets = point['acs']
                    outputs, belief = self.agent(self.curr_memory) # agent, pytorch
                    self.curr_memory['prev_belief'] = belief.detach()
                    self.curr_memory['prev_ac'] = outputs.detach()
                    valid_loss += self.criterion(outputs, targets).item() # mse loss

            points = len(self.val)*len(self.val[0])
            print(f'valid set loss: {valid_loss/points}')
            self.result_dict['val_loss']['epoch'].append(epoch + 1)
            self.result_dict['val_loss']['value'].append(valid_loss/points)
            #print(f'valid set accuracy: {valid_acc}')

    def process_data(self, length):

        
        train, val = self.train_val_split()
        self.train = []
        self.val = []
        for _ in train:
            traj = []
            for i in range(length):
                tmp_y = torch.from_numpy(_[i]["acs"].astype(float)) 
                _[i]["obs"], _[i]["acs"] = self.occlusion(_[i]), tmp_y.cuda()
                traj.append(_[i])
            self.train.append(traj)
            
        for _ in val:
            traj=[]
            for i in range(length):
                tmp_y = torch.from_numpy(_[i]["acs"].astype(float)) 
                _[i]["obs"], _[i]["acs"] = self.occlusion(_[i]), tmp_y.cuda()
                traj.append(_[i])
            self.val.append(traj)
        

    def reset_memory(self):
        self.init_state = torch.cuda.DoubleTensor(self.agent.belief_dim).fill_(0)
        self.init_ac = torch.cuda.DoubleTensor(self.acs_dim).fill_(0) 
        self.curr_ob = torch.cuda.DoubleTensor(self.obs_dim).fill_(0)
        self.curr_memory = {
        'curr_ob': self.curr_ob,    # o_t
        'prev_belief': self.init_state,   # b_{t-1}
        'prev_ac': self.init_ac,  # a_{t-1}
        'prev_ob': self.curr_ob.clone(), # o_{t-1}
        }



