import os
import torch.nn as nn
import torch
from models.FF import FF
from models.GRU import GRU

from utils import helpers as h



class GRUVAE(nn.Module):
    def __init__(self, Din, Dlatent, Dgru_hidden, Ddecoder):
        super(GRUVAE,self).__init__()
        Dgru_in = Dlatent*2
        self.gru = GRU(Dgru_in, Din, Dgru_hidden)
       
        self.ff_decoder_rec_0step = FF(Din, Dlatent, Ddecoder)
        self.ff_decoder_fwd_1step = FF(Din, Dlatent, Ddecoder)
        self.ff_decoder_bwd_1step = FF(Din, Dlatent, Ddecoder)
        self.ff_decoder_fwd_kstep = FF(Din, Dlatent, Ddecoder)
        self.ff_decoder_bwd_kstep = FF(Din, Dlatent, Ddecoder)
        

    
    def forward(self, x, hidden=None):
        x, hn = self.gru(x, hidden)
        n = int(x.size(1)/2)
        mu = x[:,:n]
        
        # variance cannot be negative and not zero for numerical stability
        sigma = nn.functional.relu(x[:,n:]) + 1e-6 * torch.ones_like(mu)
        
        # reparametrization trick: sample using gaussian standard distribution
        zero_mu = torch.zeros(mu.size()).cuda()
        zero_sigma = torch.ones(sigma.size()).cuda()
        z = mu + sigma * torch.normal(mean=zero_mu, std=zero_sigma )
        pred = {}
        pred['reconstruction'] = self.ff_decoder_rec_0step(z)
        pred['one_fwd'] = self.ff_decoder_fwd_1step(z)
        pred['one_bwd'] = self.ff_decoder_bwd_1step(z)
        pred['k_fwd'] = self.ff_decoder_fwd_kstep(z)
        pred['k_bwd'] = self.ff_decoder_bwd_kstep(z)

        if self.training:
            return pred, mu, sigma
        else:
            return pred, mu, sigma, hn

    def save(self, pth):
        torch.save(self.state_dict(), pth)




    






