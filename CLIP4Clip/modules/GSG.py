import torch
import torch.nn as nn 
from torch.nn import functional as F
from collections import OrderedDict



class GSG(nn.Module):
    def __init__(self, d,n, k, linlayer=False):
        super(GSG, self).__init__()
        #self.gate = torch.nn.Linear(d*k,k)
        self.gate = nn.Sequential(
            nn.Linear(d*k, int(d*n)),
            nn.GELU(),
            nn.Linear(int(d*n), k)
            )
        self.linlayer = linlayer
        if self.linlayer:
            self.W = nn.Sequential(*[nn.Linear(d,d, bias=False) for _ in range(k)])
    def forward(self,visual_output):
        k, b, t, d = visual_output.shape
        vis = torch.reshape(visual_output.permute(1,2,3,0), (b, t,-1))
        gate = torch.tanh(self.gate(vis))
        
        #gate = (1 + F.softmax(self.gate(vis),dim=-1))/2

        tmp = []        
        for i in range(k):
            gate_tmp = gate[:, :, i]
            tmp_vis = (gate_tmp.unsqueeze(2)*visual_output[i,:,:,:])
            if self.linlayer:
                tmp_vis = self.W[i](tmp_vis)
            tmp.append(tmp_vis)
        output = torch.stack(tmp)
        output = torch.sum(output, dim = 0)
        return output


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class Scale_Block(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, int(d_model / 2) )),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(int(d_model /2) , d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.n_head = n_head

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor):
        attn_mask_ = attn_mask.repeat_interleave(self.n_head, dim=0)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, x, attn_mask):
        # x: torch.Tensor, attn_mask: torch.Tensor
        # print(para_tuple)
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask)

