import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sklearn.neighbors import NearestNeighbors
import numpy as np

from .functionals import gem


class Flatten(nn.Module):
    def forward(self, input):
        input = input.contiguous()
        return input.view(input.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super(L2Norm, self).__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        self.work_with_tokens=work_with_tokens
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'
        
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128,
                 normalize_input=True, vladv2=False, core_loop=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
                vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(
            dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.core_loop = core_loop  # slower than non-looped, but lower memory usage

    def init_params(self, clsts, traindescs):
        if self.vladv2 == False:
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :]  # sort, descending

            self.alpha = (-np.log(0.01) /
                          np.mean(dots[0, :] - dots[1, :])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight = nn.Parameter(torch.from_numpy(
                self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            knn = NearestNeighbors(n_jobs=-1)  # TODO faiss?
            knn.fit(traindescs)
            del traindescs
            dsSq = np.square(knn.kneighbors(clsts, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) /
                          np.mean(dsSq[:, 1] - dsSq[:, 0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq

            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def forward(self, x, vis_kmaps=False, trunc_dim=None):
        N, C, H, W = x.shape

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        if vis_kmaps:
            return soft_assign.view(N, self.num_clusters, H, W)
        else:
            x_flatten = x.view(N, C, -1)

            # calculate residuals to each clusters
            vlad = torch.zeros([N, self.num_clusters, C],
                               dtype=x.dtype, layout=x.layout, device=x.device)

            if self.core_loop == True:  # slower than non-looped, but lower memory usage
                for C in range(self.num_clusters):
                    residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                        self.centroids[C:C+1, :].expand(
                            x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
                    residual *= soft_assign[:, C:C+1, :].unsqueeze(2)
                    vlad[:, C:C+1, :] = residual.sum(dim=-1)
            else:
                # calculate residuals to each clusters
                residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - self.centroids.expand(
                    x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)  # of shape N*K*C*(H*W)-1*K*C*(H*W)
                # of shape = (N*K*C*(H*W)*(N*K*1*(H*W))
                residual *= soft_assign.unsqueeze(2)
                vlad = residual.sum(dim=-1)  # of shape = N*K*C

            if trunc_dim is not None:
                vlad = vlad[:, torch.arange(vlad.size(1)) != trunc_dim]

            # intra-normalization (channel-wise)
            vlad = F.normalize(vlad, p=2, dim=2)
            vlad = vlad.view(N, -1)  # flatten  of shape N*(K*C)
            # L2 normalize       #output N*(K*C)*1 tensor
            vlad = F.normalize(vlad, p=2, dim=1)

            return vlad

class GeMPooling(nn.Module):
    def __init__(self, dim=512):
        super(GeMPooling, self).__init__()
        self.layers = nn.Sequential(
            L2Norm(),
            GeM(),
            Flatten(),
            nn.Linear(dim, dim),
            L2Norm()
        )
    def forward(self, x):
        return self.layers(x)


