import os
import torch
import torch.nn as nn
import torch.nn.functional as f
import yaml

from utils import helpers as h
import encoder_decoder as ed



def model_factory(type, obs_dim, acs_dim, configs):
    if type == "naive":
        return NaiveModel(obs_dim=obs_dim, acs_dim=acs_dim, config=configs).cuda()

    elif type == "belief":
        return BeliefModel(obs_dim=obs_dim, acs_dim=acs_dim, config=configs).cuda()


class NaiveModel(torch.nn.Module):
    def __init__(self, config, acs_dim, obs_dim):
        super(NaiveModel,self).__init__()
        self.hidden = config['actor_hidden_dim']
        self.belief_dim = config['belief_dim']
        self.policy = torch.nn.Sequential(
            nn.Linear(obs_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, acs_dim)
        )
        
        

    def forward(self, x):
        x = self.policy(x)
        return x

    def save_model(self, env_name):
        torch.save(self.state_dict(), os.path.join(self.model_dir, env_name))


class BeliefModel(torch.nn.Module):
    def __init__(self, config, acs_dim, obs_dim):
        super(BeliefModel,self).__init__()
        
        self.hidden = config['actor_hidden_dim']
        self.belief_dim = config['belief_dim']
        self.prev_acs = config['prev_acs']
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

        # self.decoder_fwd_1step = ed.get_decoder(
        #     input_dim=self.belief_dim + acs_dim,
        #     output_dim=obs_dim, 
        #     decoder_hidden=config['decoder_hidden'],
        #     nonlinearity= nn.ReLu())
        
    def forward(self, memory):   
        if self.prev_acs:
            x = torch.cat((memory['curr_ob'], memory['prev_ac']), dim=0)
        else:
            x = memory['curr_ob']
        
        prev_belief = memory['prev_belief'].double()

        #print(x.size())
        belief = self.belief_gru(x.double(), prev_belief.detach())

        pol_in = belief.squeeze()
        acs = self.policy(pol_in)
        return acs, belief
        

    def save_model(self, env_name):
        torch.save(self.state_dict(), os.path.join(self.model_dir, env_name))


class ProbBeliefModel(torch.nn.Module):
    def __init__(self, config, acs_dim, obs_dim):
        super(ProbBeliefModel,self).__init__()
        
        self.hidden = config['actor_hidden_dim']
        self.belief_dim = config['belief_dim']
        self.prev_acs = config['prev_acs']
        self.policy = torch.nn.Sequential(
            nn.Linear(self.belief_dim + self.belief_var_dim, self.hidden),
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
        self.variance_predictor = nn.Linear()
        
    def forward(self, memory):   
        if self.prev_acs:
            x = torch.cat((memory['curr_ob'], memory['prev_ac']), dim=0)
        else:
            x = memory['curr_ob']
        
        prev_belief = memory['prev_belief'].double()

        #print(x.size())
        belief = self.belief_gru(x.double(), prev_belief.detach())

        pol_in = belief.squeeze()
        acs = self.policy(pol_in)
        return acs, belief
        

    def save_model(self, env_name):
        torch.save(self.state_dict(), os.path.join(self.model_dir, env_name))





            
        

