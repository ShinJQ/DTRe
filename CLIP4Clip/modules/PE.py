import torch
from torch import nn
import torch.nn.functional as F
import math


class PE_gen(nn.Module):
    def __init__(self, pe_type, d, n, sparse=3):
        super(PE_gen, self).__init__()
        self.pe_type = pe_type
        self.sparse=sparse
        if self.pe_type == 'Learnable' or 'TRAPE':
            self.weight = nn.Sequential(
                nn.Linear(d, int(d/n)),
                nn.GELU(),
                nn.Linear(int(d/n), d)
            )


    def sine(self, visual_output):
        b, t, d = visual_output.shape
        pe = torch.zeros(t, d).to(device=visual_output.device)
        position = torch.arange(0, t).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d, 2, dtype=torch.float) * -(math.log(10000.0) / d)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.repeat(b, 1, 1)
        return pe
    
    def shift(self, pos_enc, shift_range):
        seq_len = pos_enc.size(1)
        sparse_indices = torch.arange(0, seq_len, dtype=torch.long)
        rand_shift = (torch.abs(torch.randn(seq_len))*shift_range).int().long()
        for t, i in enumerate(rand_shift):
            if t+i>=seq_len:
                rand_shift[t]=0
        sparse_indices = sparse_indices + rand_shift
        shifted_pos_enc = pos_enc[:, rand_shift, :]
        return shifted_pos_enc

    def PE_Scaler(self, pos_enc, sparse_factor):
        seq_len = pos_enc.size(1)
        sparse_indices = torch.arange(0, seq_len, step=sparse_factor, dtype=torch.long)
        tmp = [] 
        for i in range(sparse_factor):
            tmp.append(sparse_indices)
        sparse_pos = torch.sort(torch.cat(tmp)).values
        sparse_pos = sparse_pos[:seq_len]
        sparse_pos_enc = pos_enc[:, sparse_pos, :]
        return sparse_pos_enc

    def Learnable(self, visual_output):
        sine_pe = self.sine(visual_output)
        pe = self.weight(sine_pe)
        return pe


    def forward(self, visual_output, training):
        if self.pe_type == 'sine':
            pe = self.sine(visual_output)
        elif self.pe_type == 'Learnable':
            pe = self.Learnable(visual_output)
        elif self.pe_type == 'shift':
            pe = self.sine(visual_output)
            if training:
                pe = self.shift(pe, 2)
        elif self.pe_type == 'Scaled':
            pe = self.sine(visual_output)
            pe = self.PE_Scaler(pe, self.sparse)
        elif self.pe_type == 'TRAPE':
            pe = self.Learnable(visual_output)
            pe = self.PE_Scaler(pe, self.sparse)
        return pe
