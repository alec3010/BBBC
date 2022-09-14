import torch.nn as nn

class FFDO(nn.Module):
    def __init__(self, Dout, Din, hidden):
        super(FFDO,self).__init__()

        self.model = nn.Sequential(
            nn.Linear(Din, hidden),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(hidden, Dout)
            
        )
        
        

    def forward(self, x):
        x = self.model(x)
        return x

    def eval_mode(self):
        self.eval()
        