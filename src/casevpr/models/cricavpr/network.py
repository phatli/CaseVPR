import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.functional
from torch.nn.parameter import Parameter
import math


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        self.work_with_tokens = work_with_tokens

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


def gem(x, p=3, eps=1e-6, work_with_tokens=False):
    if work_with_tokens:
        x = x.permute(0, 2, 1)
        # unseqeeze to maintain compatibility with Flatten
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1./p).unsqueeze(3)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1
        return x[:, :, 0, 0]


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


class Crica(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """

    def __init__(self, dim=768):
        super().__init__()
        self.aggregation = nn.Sequential(
            L2Norm(), GeM(work_with_tokens=None), Flatten())

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=16, dim_feedforward=2048, activation="gelu", dropout=0.1)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=2)  # Cross-image encoder

    def forward(self, x):
        B, P, D = x["x_prenorm"].shape
        W = H = int(math.sqrt(P-1))
        x0 = x["x_norm_clstoken"]
        x_p = x["x_norm_patchtokens"].view(B, W, H, D).permute(0, 3, 1, 2)

        x10, x11, x12, x13 = self.aggregation(x_p[:, :, 0:8, 0:8]), self.aggregation(
            x_p[:, :, 0:8, 8:]), self.aggregation(x_p[:, :, 8:, 0:8]), self.aggregation(x_p[:, :, 8:, 8:])
        x20, x21, x22, x23, x24, x25, x26, x27, x28 = self.aggregation(x_p[:, :, 0:5, 0:5]), self.aggregation(x_p[:, :, 0:5, 5:11]), self.aggregation(x_p[:, :, 0:5, 11:]), \
            self.aggregation(x_p[:, :, 5:11, 0:5]), self.aggregation(x_p[:, :, 5:11, 5:11]), self.aggregation(x_p[:, :, 5:11, 11:]), \
            self.aggregation(x_p[:, :, 11:, 0:5]), self.aggregation(
                x_p[:, :, 11:, 5:11]), self.aggregation(x_p[:, :, 11:, 11:])
        x = [i.unsqueeze(1) for i in [x0, x10, x11, x12, x13,
                                      x20, x21, x22, x23, x24, x25, x26, x27, x28]]

        x = torch.cat(x, dim=1)
        x = self.encoder(x).view(B, 14*D)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x
