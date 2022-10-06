import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, Dout, Din, Dhidden):
        super(GRU,self).__init__()

                
       

        self.gru = nn.GRU(input_size=Din, 
                                       hidden_size=Dhidden, 
                                       num_layers=1,
                                       batch_first=True)
        self.mu_fc = nn.Linear(Dhidden, Dout)
        self.hidden = None
        

        
    def forward(self, x):
        belief, hn = self.gru(x, self.hidden)
        mu = self.mu_fc(belief)
        
        return mu
        
            
        
        
