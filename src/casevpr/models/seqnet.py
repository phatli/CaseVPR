import torch
import torch.nn as nn
import torch.nn.functional as F

class seqNet(nn.Module):
    def __init__(self, inDims, outDims, seqL, w=3):

        super(seqNet, self).__init__()
        self.inDims = inDims
        self.outDims = outDims
        self.w = w
        self.conv = nn.Conv1d(inDims, outDims, kernel_size=self.w)

    def forward(self, x):
        
        if len(x.shape) < 3:
            x = x.unsqueeze(1) # convert [B,C] to [B,1,C]

        x = x.permute(0,2,1) # from [B,T,C] to [B,C,T]
        seqFt = self.conv(x)
        seqFt = torch.mean(seqFt,-1)

        seqFt = seqFt.view(seqFt.size(0),-1)
        seqFt = F.normalize(seqFt, p=2, dim=1)

        return seqFt