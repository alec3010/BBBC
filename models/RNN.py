import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, Dout, Din, Dhidden):
        super(RNN,self).__init__()

        self.fc = nn.Linear(Dhidden, Dout)        
      

        self.belief_gru = nn.GRU(input_size=Din, 
                                       hidden_size=Dhidden, 
                                       num_layers=1,
                                       batch_first=True)
        

        
    def forward(self, x, hidden=None):
        x, hn = self.belief_gru(x, hidden)
        x = self.fc(x)
        return x, hn
        
            
        
        
