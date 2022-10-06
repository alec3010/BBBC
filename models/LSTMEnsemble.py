import torch.nn as nn
from models.LSTM import LSTM

class LSTMEnsemble(nn.Module):

    def __init__(self, Dout, Din, Dhidden):
        super(LSTMEnsemble, self).__init__()

        self.lstm1 = LSTM(Dout, Din, Dhidden)
        self.lstm2 = LSTM(Dout, Din, Dhidden)
        self.lstm3 = LSTM(Dout, Din, Dhidden)
        self.lstm4 = LSTM(Dout, Din, Dhidden)
        self.lstm5 = LSTM(Dout, Din, Dhidden)

    def initialize_hidden_state(self):
        self.lstm1.hidden_state = None
        self.lstm2.hidden_state = None
        self.lstm3.hidden_state = None
        self.lstm4.hidden_state = None
        self.lstm5.hidden_state = None

    def forward(self, x):
        mu1 = self.lstm1(x)
        mu2 = self.lstm2(x)
        mu3 = self.lstm3(x)
        mu4 = self.lstm4(x)
        mu5 = self.lstm5(x)
        mu = mu1 + mu2 + mu3 + mu4 + mu5
        mean = mu / 5.0
        return mean, [mu1, mu2, mu3, mu4, mu5]