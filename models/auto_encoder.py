import os
from models.GRUEnsemble import GRUEnsemble
from models.LSTMEnsemble import LSTMEnsemble
import torch
import torch.nn as nn
import torch.nn.functional as f
from models.FF import FF
from models.FFDO import FFDO

from utils import helpers as h



class AutoEncoder(nn.Module):
    def __init__(self, Din, Dacs, Dacsencoding, Dobs, Dlatent, Dgru_hidden, Ddecoder, k, acs_feedback):
        super(AutoEncoder,self).__init__()
        self.testing = False
        self.acs_feedback = acs_feedback
        if self.acs_feedback:
            self.gru_ensemble = GRUEnsemble(Dlatent, Dacs+Dobs, Dgru_hidden)
        else:
            self.gru_ensemble = GRUEnsemble(Dlatent, Dobs, Dgru_hidden)
        
        kernel_size = 3
        stride = padding = 1
        num_filters1 = num_filters2 = 5

        self.conv1_future_seq = nn.Conv1d(k + 1, num_filters1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2_future_seq = nn.Conv1d(num_filters1, num_filters2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.convLinear_future = nn.Linear(num_filters2*Dacs, Dacsencoding)

        self.conv1_past_seq = nn.Conv1d(k + 1, num_filters1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2_past_seq = nn.Conv1d(num_filters1, num_filters2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.convLinear_past = nn.Linear(num_filters2*Dacs, Dacsencoding)
        
        
        self.ff_decoder_rec = FFDO(Dobs, Dlatent + Dacs, Ddecoder)
        self.ff_decoder_fwd_1step = FFDO(Dobs, Dlatent + Dacs, Ddecoder)
        self.ff_decoder_bwd_1step = FFDO(Dobs, Dlatent + Dacs, Ddecoder)
        self.ff_decoder_fwd_kstep = FFDO(Dobs, Dlatent + Dacsencoding, Ddecoder)
        self.ff_decoder_bwd_kstep = FFDO(Dobs, Dlatent + Dacsencoding, Ddecoder)
        self.ff_decoder_acs_1step = FFDO(Dacs, Dlatent + Dobs, Ddecoder)
        # self.ff_decoder_acs_kstep = FFDO(Dacs*(k+1), Dlatent + Dobs, Ddecoder)
        

    
    def forward(self, obs, acs, future_acs = None, past_acs = None):

        
        if self.acs_feedback:
            input_ = torch.cat((obs, acs), -1)
        else:  
            input_ = obs
       
        
        mu, mu_list = self.gru_ensemble(input_)
        if not self.testing:
    
            padded_mu = h.zero_pad(mu)

            pred = {}

            # reconstruction: (b_{t}, a_{t-1} -> o_t)
            pred['reconstruction'] = self.ff_decoder_rec(torch.cat((mu, acs), dim=1))

            # 1-step forward prediction: (b_{t-1}, a_{t-1} -> o_t)
            pred['obs_one_fwd'] = self.ff_decoder_fwd_1step(torch.cat((padded_mu[:-1], acs), dim=1) ) # 
            
            # 1-step backward prediction (b_t, a_{t-1} -> o_{t-1})
            pred['obs_one_bwd'] = self.ff_decoder_bwd_1step(torch.cat((mu, acs), dim=1)) # 
            
            # k-step forward prediciton: (b_{t-1}, a_{t-1}:a_{t+k-1} -> o_{t+k})
            x = f.relu(self.conv1_future_seq(future_acs))
            x = f.relu(self.conv2_future_seq(x))
            encoded_future_acs = self.convLinear_future(x.view(x.size(0), -1))

            pred['obs_k_fwd'] = self.ff_decoder_fwd_kstep(torch.cat((padded_mu[:-1], encoded_future_acs), dim=1)) # 
            
            # k-step backward prediction: (b_{t}, a_{t-1}:a_{t-k-1} -> o_{t-k-1})
            x = f.relu(self.conv1_past_seq(past_acs))
            x = f.relu(self.conv2_past_seq(x))
            encoded_past_acs = self.convLinear_past(x.view(x.size(0), -1))

            pred['obs_k_bwd'] = self.ff_decoder_bwd_kstep(torch.cat((mu, encoded_past_acs), dim=1)) # 

            # 1-step action prediction (b_{t-1}, o_t -> a_{t-1})
            pred['acs_1_fwd'] = self.ff_decoder_acs_1step(torch.cat((padded_mu[:-1], obs), dim=1 )) # 

            # k-step action prediction 
            # pred['acs_k_fwd'] = self.ff_decoder_acs_kstep(torch.cat((padded_mu[:-1], obs[2*k+1:]), dim=1 ))

        if self.training:
            return pred, mu, mu_list
        elif self.testing:
            return mu, mu_list
        else:
            return pred, mu, mu_list
        

    def save(self, pth):
        torch.save(self.state_dict(), pth)

    def test(self):
        if self.testing:
            self.testing = False
        else:
            self.testing = True




    






