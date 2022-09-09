from doctest import testfile
import numpy as np

from utils import data_loader 
from utils import helpers as h

if __name__ == "__main__":

    _, _, x, y = h.train_val_split("data/LQRKalman_demo_InvertedPendulum.pickle", 0.9)
 
    loader = data_loader.DataLoader(x, y, 500, [0,1], "RNNVAE")
    print("Length is", len(loader))
    print("Loader contains {} sequences of data".format(loader.sequence_number))
    

    for (x, y) in loader:
        print('next obs size:', x_n.size)
        assert x.device==y.device, "Labels and Data not on same device"
        print('obs size:', x.size())
        # assert x.size()[0] == y.size()[0], "Batch Sizes do not fit"
        # if x.size()[1] != y.size()[1]:
        #     print(x.size()[1])
        #     print(y.size()[1])
        # assert x.size()[1] == y.size()[1], "Sequence lengths do not fit"


