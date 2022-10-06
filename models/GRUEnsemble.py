import torch.nn as nn
from models.GRU import GRU

class GRUEnsemble(nn.Module):

    def __init__(self, Dout, Din, Dhidden):
        super(GRUEnsemble, self).__init__()

        self.gru1 = GRU(Dout, Din, Dhidden)
        self.gru2 = GRU(Dout, Din, Dhidden)
        self.gru3 = GRU(Dout, Din, Dhidden)
        self.gru4 = GRU(Dout, Din, Dhidden)
        self.gru5 = GRU(Dout, Din, Dhidden)

    def initialize_hidden_state(self):
        self.gru1.hidden_state = None
        self.gru2.hidden_state = None
        self.gru3.hidden_state = None
        self.gru4.hidden_state = None
        self.gru5.hidden_state = None

    def forward(self, x):
        mu1 = self.gru1(x)
        mu2 = self.gru2(x)
        mu3 = self.gru3(x)
        mu4 = self.gru4(x)
        mu5 = self.gru5(x)
        mu = mu1 + mu2 + mu3 + mu4 + mu5
        mean = mu / 5.0
        return mean, [mu1, mu2, mu3, mu4, mu5]