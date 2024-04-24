import torch.nn as nn

class AddNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(AddNorm, self).__init__()
        self.size = size
        self.norm = nn.LayerNorm(size)
        
    
    def forward(self, x, sublayer):
        return x + self.norm(x + sublayer)
    
