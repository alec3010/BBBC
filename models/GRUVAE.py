import os
import torch.nn as nn
import torch
from models.FF import FF
from models.FFDO import FFDO
from models.GRU import GRU

from utils import helpers as h



class GRUVAE(nn.Module):
    def __init__(self, Din, Dacs, Dlatent, Dgru_hidden, Ddecoder):
        super(GRUVAE,self).__init__()
        self.gru = GRU(Dlatent, Din, Dgru_hidden)


       
        self.ff_decoder_rec_0step = FFDO(Din, Dlatent, Ddecoder)
        self.ff_decoder_fwd_1step = FFDO(Din, Dlatent, Ddecoder)
        self.ff_decoder_bwd_1step = FFDO(Din, Dlatent, Ddecoder)
        self.ff_decoder_fwd_kstep = FFDO(Din, Dlatent, Ddecoder)
        self.ff_decoder_bwd_kstep = FFDO(Din, Dlatent, Ddecoder)
        self.ff_decoder_acs_1step = FFDO(Dacs, Dlatent, Ddecoder)
        

    
    def forward(self, x, hidden=None):
        mu, log_sigma, hn = self.gru(x, hidden)
        
        # reparametrization trick: sample using gaussian standard distribution
        epsilon = torch.randn(mu.shape).cuda()
        z = mu + epsilon*torch.exp(0.5 * log_sigma )
        
        pred = {}
        pred['reconstruction'] = self.ff_decoder_rec_0step(z)
        pred['one_fwd'] = self.ff_decoder_fwd_1step(z)
        pred['one_bwd'] = self.ff_decoder_bwd_1step(z)
        pred['k_fwd'] = self.ff_decoder_fwd_kstep(z)
        pred['k_bwd'] = self.ff_decoder_bwd_kstep(z)
        pred['acs'] = self.ff_decoder_acs_1step(mu)
        # sigma = torch.exp(log_sigma)

        if self.training:
            return pred, mu, log_sigma
        else:
            return pred, mu, log_sigma, hn

    def save(self, pth):
        torch.save(self.state_dict(), pth)




    






