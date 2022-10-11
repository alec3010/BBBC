import os
from models.GRUEnsemble import GRUEnsemble
from models.LSTMEnsemble import LSTMEnsemble
import torch
import torch.nn as nn
import torch.nn.functional as f
from models.Decoder import Decoder

from utils import helpers as h



class AutoEncoder(nn.Module):
    def __init__(self, Din, Dacs, Dacsencoding, Dobs, Dlatent, Dgru_hidden, Ddecoder, k, acs_feedback):
        super(AutoEncoder,self).__init__()
        self.testing = False
        self.acs_feedback = acs_feedback
        if self.acs_feedback:
            self.encoder = LSTMEnsemble(Dlatent, Dacs+Dobs, Dgru_hidden)
        else:
            self.encoder = LSTMEnsemble(Dlatent, Dobs, Dgru_hidden)
        
        
        
        self.ff_decoder_rec = Decoder (Dobs, Dlatent, Ddecoder)
        self.ff_decoder_fwd_1step = Decoder (Dobs, Dlatent + Dacs, Ddecoder)
        self.ff_decoder_bwd_1step = Decoder (Dobs, Dlatent + Dacs, Ddecoder)
        self.ff_decoder_fwd_kstep = Decoder (Dobs, Dlatent + k*Dacs, Ddecoder)
        self.ff_decoder_bwd_kstep = Decoder (Dobs, Dlatent + k*Dacs, Ddecoder)
        self.ff_decoder_acs_1step = Decoder (Dacs, Dlatent, Ddecoder)
        # self.ff_decoder_acs_kstep = Decoder DO(Dacs*(k+1), Dlatent + Dobs, Ddecoder)
        

    
    def forward(self, obs, acs, future_acs = None, past_acs = None):

        
        if self.acs_feedback:
            input_ = torch.cat((obs, acs), -1)
        else:  
            input_ = obs
       
        
        mu, mu_list = self.encoder(input_)
        if not self.testing:
    
            padded_mu = h.zero_pad(mu)

            pred = {}

            # reconstruction: (b_{t}, a_{t-1} -> o_t)
            pred['reconstruction'] = self.ff_decoder_rec(mu)#torch.cat((mu, acs), dim=1)

            # # 1-step forward prediction: (b_{t-1}, a_{t-1} -> o_t)
            # pred['obs_one_fwd'] = self.ff_decoder_fwd_1step( #torch.cat((mu, acs), dim=1)) # 
            
            # # 1-step backward prediction (b_t, a_{t-1} -> o_{t-1})
            # pred['obs_one_bwd'] = self.ff_decoder_bwd_1step()#torch.cat((mu, acs), dim=1) # 
            
            # k-step forward prediciton: (b_{t-1}, a_{t-1}:a_{t+k-1} -> o_{t+k})
            pred['obs_k_fwd'] = self.ff_decoder_fwd_kstep(torch.cat((mu, future_acs), dim=1)) # 
            
            # k-step backward prediction: (b_{t}, a_{t-1}:a_{t-k} -> o_{t-k})

            pred['obs_k_bwd'] = self.ff_decoder_bwd_kstep(torch.cat((mu, past_acs), dim=1)) # 

            # 1-step action prediction (b_{t-1}, o_t -> a_{t-1})
            # pred['acs_1_fwd'] = self.ff_decoder_acs_1step(torch.cat((mu), dim=1 )) # 

            # k-step action prediction 
            # pred['acs_k_fwd'] = self.ff_decoder_acs_kstep(torch.cat((mu, obs[2*k+1:]), dim=1 ))

        
        if self.testing:
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
    
    def init_hidden(self):
        self.encoder.initialize_hidden_state()




    






