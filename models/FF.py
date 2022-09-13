import torch.nn as nn

class FF(nn.Module):
    def __init__(self, Dout, Din, hidden):
        super(FF,self).__init__()

        self.model = nn.Sequential(
            nn.Linear(Din, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, Dout)
        )
        
        

    def forward(self, x):
        x = self.model(x)
        return x

    def eval_mode(self):
        self.eval()
        