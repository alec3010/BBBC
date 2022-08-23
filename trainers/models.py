import torch
import torch.nn as nn

class MDPModel(torch.nn.Module):
    def __init__(self, policy_input_dim, actor_hidden_dim, action_dim):
        super(MDPModel,self).__init__()
        self.policy = torch.nn.Sequential(
            nn.Linear(policy_input_dim , actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, action_dim)
        )
        
        
    def forward(self,x):
        x = self.policy(x)
        
        return x
        


class POMDPModel(MDPModel):
    def __init__(self, belief_dim, obs_dim, actor_hidden_dim, action_dim):
        super(POMDPModel, self).__init__(policy_input_dim=belief_dim, actor_hidden_dim=actor_hidden_dim, action_dim=action_dim)
        
        self.policy = self.policy.double()
        self.belief_gru = torch.nn.GRUCell(obs_dim + action_dim, belief_dim)
        self.belief_gru = self.belief_gru.double()
        
    def forward(self, memory):
               
        x = torch.cat((memory['curr_ob'], memory['prev_ac']), dim=0)
        prev_belief = memory['prev_belief'].double()
        belief = self.belief_gru(x.double(), prev_belief.detach())

        pol_in = belief.squeeze()
        acs = self.policy(pol_in)
        return acs, belief



            
        

