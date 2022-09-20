import numpy as np
from typing import Iterable
from dataclasses import dataclass
import torch
from utils import helpers as h

    

class DataLoader():
    
    def __init__(self, x, y, seq_length, batch_size, idx_list, arch) -> None:

        assert len(x)==len(y), "Val and Train must have the same lengths!"
        assert len(x)>seq_length, "Sequence length must be shorter than length of dataset!"
        assert arch in {"FF", "GRUVAE"}, "Invalid Architecture!"
        
        self.x = x
        self.y = y
        self.seq_length = seq_length
        self.idx_list = idx_list
        self.arch = arch
        self.idxs = set()
        self.load_sequences(seq_length=seq_length)



    def __iter__(self):
        if self.arch =="GRUVAE":
            return zip(self.batches_x, self.batches_y)
        if self.arch =="FF":
            return zip(self.sequences_x, self.sequences_y)


    def __len__(self):
        '''
        returns length of usable data
        '''
        return (len(self.x) - self.seq_length)
    
    def len(self):
        '''
        returns amount of sequences
        '''
        len_ = int((len(self.x) - self.seq_length)/self.seq_length)
        return len_

    def load_sequences(self, seq_length=30):
        # indexes of the sections in the dataset that were already used to create a batch
        self.sequences_x = []
        self.sequences_y = []
        self.seq_amount_ = 0
        done = False
        nr = 1
        while not done:
            s_x, s_y = self.sample_sequence(seq_length=seq_length)
            self.sequences_x.append(s_x)
            self.sequences_y.append(s_y)
            done = len(self.idxs)==self.len()
            nr += 1
            #print(nr)
        
        assert len(self.sequences_x) == len(self.sequences_y), "Sequence Lengths are different"
      

        self.sequence_number = len(self.sequences_x)

    def load_batches(self, seq_length=30, batch_size=32):
        # indexes of the sections in the dataset that were already used to create a batch
        self.batches_x = []
        self.batches_y = []
        self.seq_amount_ = 0
        done = False
        nr = 1
        print("Sampling batches")
        while not done:
            b_x, b_y = self.sample_batch(batch_size=batch_size, seq_length=seq_length)
            self.batches_x.append(b_x)
            self.batches_y.append(b_y)
            done = len(self.idxs)==self.len()
            nr += 1
        
        assert len(self.batches_x) == len(self.batches_y), "Sequence Lengths are different"
      

        self.batch_number = len(self.batches_x)
        
    
    def sample_sequence(self, seq_length):
        '''
        Samples sequence of length <seq_length> of data from trajectory, making sure that every data point is only used once
        '''
        s_x_ = []
        s_y_ = []

        done = False
        idx = 0
        while not done:
            
            idx = np.random.randint(0, int((len(self.x) - seq_length)/seq_length))
            if idx in self.idxs:
                continue
            else:
                self.idxs.add(idx)
                done = True

        for i in range(seq_length):

            # occlusion depending on selected process model
            tmp = []
            for j in self.idx_list:
                tmp.append(self.x[idx+i][j].item())
            tmp_x = torch.cuda.FloatTensor(tmp)
            tmp_y = torch.cuda.FloatTensor(self.y[idx+i])
            s_x_.append(tmp_x)
            s_y_.append(tmp_y.cuda())
            
        s_x = torch.stack(s_x_, dim=0)
        s_y = torch.stack(s_y_, dim=0)
                    
        self.seq_amount_ += 1
        return s_x, s_y

    def sample_batch(self, batch_size, seq_length):
        print('Sampling batch')
        b_x_ = []
        b_y_ = []
        done = False
        for i in range(batch_size):
            
            print("adding sequence to batch")
            s_x, s_y = self.sample_sequence(seq_length=seq_length)
            b_x_.append(s_x)
            b_y_.append(s_y)
            done = len(self.idxs)==self.len()
            if done: break
        
        
        b_x = torch.stack(b_x_, dim=0)
        b_y = torch.stack(b_y_, dim=0)

        return b_x, b_y


        
        
            
            
   
            
        # if self.arch == "FF":
            # 
            # 
        # elif self.arch =="RNNVAE":
            # point_y = y_[-1]


        
        