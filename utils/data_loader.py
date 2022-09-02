import numpy as np
from typing import Iterable
from dataclasses import dataclass
import torch
from utils import helpers as h

    

class DataLoader():
    
    def __init__(self, x, y, seq_length, batch_size, idx_list, arch) -> None:

        assert len(x)==len(y), "Val and Train must have the same lengths!"
        assert len(x)>seq_length, "Sequence length must be shorter than length of dataset!"
        assert arch in {"FF", "RNNFF"}, "Invalid Architecture!"
        


        self.x = x
        self.y = y
        
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.idx_list = idx_list
        self.arch = arch
        self.load_batches()
        pass



    def __iter__(self):
        return zip(self.batches_x, self.batches_y)


    def __len__(self):
        '''
        returns length of usable data
        '''
        
        return self.len_
    
    def len(self):
        '''
        returns length of usable data
        '''
        len_ = len(self.x) - self.seq_length
        return len_

    def load_batches(self, seq_length=30, batch_size=32):
        self.idxs = set() # indexes of the sections in the dataset that were already used to create a batch
        self.batches_x = []
        self.batches_y = []
        self.len_ = 0
        done = False
        nr = 1
        while not done:
            b_x, b_y, idx = self.sample_batch(seq_length=seq_length, batch_size=batch_size)
            # print(len(self.idxs))
            # print(len(self))
            if (self.len()-len(self.idxs))<seq_length:
                print("Created all possible batches")
                done=True
            self.batches_x.append(b_x)
            self.batches_y.append(b_y)
            nr += 1
        assert len(self.batches_x) == len(self.batches_y), "Batch Lengths are different"

        self.batch_number = len(self.batches_x)
        
    
    def sample_batch(self, seq_length, batch_size):
        '''
        Samples <batch_size> random sequences of length <seq_length> of data from trajectory
        '''
        done = False
        b_x_ = []
        b_y_ = []

        for _ in range(batch_size):
            idx_found = False
            while not idx_found:
                idx = np.random.randint(0, len(self.x) - seq_length)
                if idx in self.idxs:
                    continue
                else:
                    
                    self.idxs.add(idx)
                    idx_found = True
                    

            
            x_ = []
            y_ = []

            for i in range(seq_length):
                # occlusion depending on selected process model
                tmp = []
                for j in self.idx_list:
                    tmp.append(self.x[idx+i][j].item())
                tmp_x = torch.cuda.FloatTensor(tmp)
                tmp_y = torch.from_numpy(self.y[idx+i].astype(float))
                x_.append(tmp_x)
                y_.append(tmp_y.cuda())
            
            point_x = torch.stack(x_, dim=0)
            
            if self.arch == "FF":
                point_y = torch.stack(y_, dim=0)
                
            elif self.arch =="RNNFF":
                point_y = y_[-1]

            self.len_ += 1
            
            
            b_x_.append(point_x.cuda())
            b_y_.append(point_y.cuda())
        b_x = torch.stack(b_x_, dim=0)
        b_y = torch.stack(b_y_, dim=0)


        
        return b_x, b_y, idx