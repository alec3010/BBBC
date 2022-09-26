import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, Dout, Din, Dhidden):
        super(GRU,self).__init__()

        self.mu_fc = nn.Linear(Dhidden, Dout)        
        self.sigma_fc = nn.Linear(Dhidden, Dout)

        self.gru = nn.GRU(input_size=Din, 
                                       hidden_size=Dhidden, 
                                       num_layers=1,
                                       batch_first=True)
        

        
    def forward(self, x, hidden=None):
        x, hn = self.gru(x, hidden)
        mu = self.mu_fc(x)
        sigma = self.sigma_fc(x)
        return mu, sigma, hn
        
            
        
        
