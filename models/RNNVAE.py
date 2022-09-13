import os
import torch.nn as nn
import torch
from models.FF import FF
from models.RNN import RNN

from utils import helpers as h



class RNNVAE(nn.Module):
    def __init__(self, config, acs_dim, obs_dim):
        super(RNNVAE,self).__init__()
        
        self.hidden_actor = config['actor_hidden_dim']
        self.belief_dim = config['belief_dim'] 
        self.rnn_hidden = config['hidden_dim']
        self.decoder_hidden = config['decoder_hidden']
        self.task_agnostic = config['task_agnostic']
        self.belief_reg = config['belief_reg']
        
        self.acs_dim = acs_dim
        self.rnn = RNN(self.belief_dim*2, obs_dim, self.rnn_hidden)
        
        self.ff_decoder_rec_0step = FF(obs_dim, self.belief_dim, self.decoder_hidden)
        self.ff_decoder_fwd_1step = FF(obs_dim, self.belief_dim, self.decoder_hidden)
        self.ff_decoder_bwd_1step = FF(obs_dim, self.belief_dim, self.decoder_hidden)
        self.ff_decoder_fwd_kstep = FF(obs_dim, self.belief_dim, self.decoder_hidden)
        self.ff_decoder_bwd_kstep = FF(obs_dim, self.belief_dim, self.decoder_hidden)
        

    
    def forward(self, x, hidden=None):

        
        # the subscript s denoting that mu and sigma refer to the true state
        mu_s, sigma_s, hn = self.gaussian_RNN_encoder(x, hidden)
        # reparametrization trick: sample using gaussian standard distribution
        zero_mu = torch.zeros(mu_s.size()).cuda()
        zero_sigma = torch.ones(sigma_s.size()).cuda()
        z = mu_s + sigma_s * torch.normal(mean = zero_mu, std= zero_sigma )

        pred = {}
        pred['reconstruction'] = self.ff_decoder_rec_0step(z)
        pred['one_fwd'] = self.ff_decoder_fwd_1step(z)
        pred['one_bwd'] = self.ff_decoder_bwd_1step(z)
        pred['k_fwd'] = self.ff_decoder_fwd_kstep(z)
        pred['k_bwd'] = self.ff_decoder_bwd_kstep(z)

        if self.training:
            return pred, mu_s, sigma_s
        else:
            
            return pred, mu_s, sigma_s, hn

    def gaussian_RNN_encoder(self, x, hidden):
        n = self.belief_dim
        x, hn = self.rnn(x, hidden)
        mu = x[:,:n]
        
        # variance cannot be negative and not zero for numerical stability
        sigma = nn.functional.relu(x[:,n:]) + 1e-6 * torch.ones_like(mu)
        return mu, sigma, hn



    def save_model(self, env_name):
        torch.save(self.state_dict(), os.path.join(self.model_dir, env_name))







