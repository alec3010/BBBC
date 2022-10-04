import os
import torch
import torch.nn as nn
import torch.nn.functional as f
from models.FF import FF
from models.FFDO import FFDO
from models.GRU import GRU

from utils import helpers as h



class GRUVAE(nn.Module):
    def __init__(self, Din, Dacs, Dacsencoding, Dobs, Dlatent, Dgru_hidden, Ddecoder, k, acs_feedback):
        super(GRUVAE,self).__init__()
        self.testing = False
        self.acs_feedback = acs_feedback
        if self.acs_feedback:
            self.gru = GRU(Dlatent, Dacs+Dobs, Dgru_hidden)
        else:
            self.gru = GRU(Dlatent, Dobs, Dgru_hidden)
        
        kernel_size = 3
        stride = padding = 1
        num_filters1 = num_filters2 = 5

        self.conv1_future_seq = nn.Conv1d(k + 1, num_filters1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2_future_seq = nn.Conv1d(num_filters1, num_filters2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.convLinear_future = nn.Linear(num_filters2*Dacs, Dacsencoding)

        self.conv1_past_seq = nn.Conv1d(k + 1, num_filters1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2_past_seq = nn.Conv1d(num_filters1, num_filters2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.convLinear_past = nn.Linear(num_filters2*Dacs, Dacsencoding)
        
   
        self.ff_decoder_fwd_1step = FFDO(Dobs, Dlatent + Dacs, Ddecoder)
        self.ff_decoder_bwd_1step = FFDO(Dobs, Dlatent + Dacs, Ddecoder)
        self.ff_decoder_fwd_kstep = FFDO(Dobs, Dlatent + Dacsencoding, Ddecoder)
        self.ff_decoder_bwd_kstep = FFDO(Dobs, Dlatent + Dacsencoding, Ddecoder)
        self.ff_decoder_acs_1step = FFDO(Dacs, Dlatent + Dobs, Ddecoder)
        self.ff_decoder_acs_kstep = FFDO(Dacs*(k+1), Dlatent + Dobs, Ddecoder)
        

    
    def forward(self, obs, acs, k, hidden=None):

        
        if self.acs_feedback and not self.testing:
            input_ = torch.cat((obs[k+1:-k], acs[k+1:-k]), -1)
        elif not self.acs_feedback and not self.testing: 
            input_ = obs[k+1:-k-1]
        elif not self.acs_feedback and self.testing:
            input_ = obs
        elif self.acs_feedback and self.testing:
            input_ = torch.cat((obs, acs), -1)
        
        mu, log_sigma, hn = self.gru(input_, hidden)


       
        if not self.testing:
            future_acs = h.tm1_tpkm1(acs, k)
            past_acs = h.tmkm1_tm1(acs, k)
            # reparametrization trick: sample using gaussian standard distribution
            epsilon = torch.randn(mu.shape).cuda()
            z = mu + epsilon*torch.exp(0.5 * log_sigma )

        
            padded_z = h.zero_pad(z)
            padded_mu = h.zero_pad(mu)
   
            pred = {}

            # 1-step forward prediction: (b_{t-1}, a_{t-1} -> o_t)
            pred['obs_one_fwd'] = self.ff_decoder_fwd_1step(torch.cat((padded_z[:-1], acs[k:-k-1,:]), dim=1))
            
            # 1-step backward prediction (b_t, a_{t-1} -> o_{t-1})
            pred['obs_one_bwd'] = self.ff_decoder_bwd_1step(torch.cat((z, acs[k:-k-1,:]), dim=1))
            
            # k-step forward prediciton: (b_{t-1}, a_{t-1}:a_{t+k-1} -> o_{t+k})
            x = f.relu(self.conv1_future_seq(future_acs))
            x = f.relu(self.conv2_future_seq(x))
            encoded_future_acs = self.convLinear_future(x.view(x.size(0), -1))
            pred['obs_k_fwd'] = self.ff_decoder_fwd_kstep(torch.cat((padded_z[:-1], encoded_future_acs), dim=1))

            # k-step backward prediction: (b_{t}, a_{t-1}:a_{t-k-1} -> o_{t-k-1})
            x = f.relu(self.conv1_past_seq(past_acs))
            x = f.relu(self.conv2_past_seq(x))
            encoded_past_acs = self.convLinear_past(x.view(x.size(0), -1))
            pred['obs_k_bwd'] = self.ff_decoder_bwd_kstep(torch.cat((z, encoded_past_acs), dim=1))

            # 1-step action prediction (b_{t}, o_t -> a_{t-1})
            pred['acs_1_fwd'] = self.ff_decoder_acs_1step(torch.cat((padded_mu[:-1], obs[k+1:-k]), dim=1 ))

            # k-step action prediction 
            pred['acs_k_fwd'] = self.ff_decoder_acs_kstep(torch.cat((padded_mu[:-1], obs[2*k+1:]), dim=1 ))
            # sigma = torch.exp(log_sigma)

        if not self.testing and self.training:
            return pred, mu, log_sigma
        elif not self.testing:
            return pred, mu, log_sigma, hn
        else:
            return mu, log_sigma, hn

    def save(self, pth):
        torch.save(self.state_dict(), pth)

    def test(self):
        if self.testing:
            self.testing = False
        else:
            self.testing = True




    






