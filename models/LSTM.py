import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, Dout, Din, Dhidden):
        super(LSTM,self).__init__()
        self.gru = nn.LSTM(input_size=Din, 
                                       hidden_size=Dhidden, 
                                       num_layers=1,
                                       batch_first=True)
        self.mu_fc = nn.Linear(Dhidden, Dout)
        self.hidden = None
        

        
    def forward(self, x):
        belief, (hn, cn) = self.gru(x, self.hidden)
        mu = self.mu_fc(belief)
        
        return mu
        
            
        
        