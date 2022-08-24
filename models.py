import os
import torch
import torch.nn as nn
import torch.nn.functional as f
import yaml

from utils import helpers as h



def model_factory(type, obs_dim, acs_dim):
    if type == "naive":
        return NaiveModel(obs_dim=obs_dim, acs_dim=acs_dim).cuda()

    elif type == "belief":
        return BeliefModel(obs_dim=obs_dim, acs_dim=acs_dim).cuda()


class NaiveModel(torch.nn.Module):
    def __init__(self, obs_dim, acs_dim):
        super(NaiveModel,self).__init__()
        self.get_params()
        self.obs_dim = obs_dim
        self.acs_dim = acs_dim
        self.policy = torch.nn.Sequential(
            nn.Linear(obs_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, acs_dim)
        )
        
        
        
    def get_params(self):
        config = h.get_params("./configs/learning_params.yaml")
        self.hidden = config['actor_hidden_dim']
        self.belief_dim = config['belief_dim']
        self.model_dir = config['model_dir']
        

    def forward(self, x):
        x = self.policy(x)
        x = 3*torch.tanh(x)
        return x

    def save_model(self, env_name):
        torch.save(self.state_dict(), os.path.join(self.model_dir, env_name))


class BeliefModel(torch.nn.Module):
    def __init__(self, obs_dim, acs_dim):
        super(BeliefModel,self).__init__()
        self.get_params()
        self.policy = torch.nn.Sequential(
            nn.Linear(self.belief_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, acs_dim)
        )
        
        self.policy = self.policy.double()
        if self.prev_acs:
            self.belief_gru = torch.nn.GRUCell(obs_dim + acs_dim, self.belief_dim)
        else:
            self.belief_gru = torch.nn.GRUCell(obs_dim, self.belief_dim)
        self.belief_gru = self.belief_gru.double()
        
    def forward(self, memory):   
        if self.prev_acs:
            x = torch.cat((memory['curr_ob'], memory['prev_ac']), dim=0)
        else:
            x = memory['curr_ob']
        prev_belief = memory['prev_belief'].double()

        
        belief = self.belief_gru(x.double(), prev_belief.detach())

        pol_in = belief.squeeze()
        acs = self.policy(pol_in)
        acs = 3*torch.tanh(acs)
        return acs, belief

    def get_params(self):
        config = h.get_params("./configs/learning_params.yaml")
        self.hidden = config['actor_hidden_dim']
        self.belief_dim = config['belief_dim']
        self.model_dir = config['model_dir']
        self.prev_acs = config['prev_acs']

    def save_model(self, env_name):
        torch.save(self.state_dict(), os.path.join(self.model_dir, env_name))




            
        
