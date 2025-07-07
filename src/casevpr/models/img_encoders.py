import torch
import torch.nn as nn
import numpy as np

from .cricavpr.network import Crica
from .backbone import load_basic_model
from .nn_module import Flatten, L2Norm

# Create a pseudo class as imgVPR that reply random vectors
class imgDonothing(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.random.rand(30, 40), torch.randn(1, 4096)

class imgVPR(nn.Module):
    def __init__(self, img_model_name="crica", arch = "dinov2", pooling = "crica", pool_class = Crica, pool_class_args = {}):
        super().__init__()
        self.encoder, encoder_dim = load_basic_model(arch)
        self.arch = arch
        pool_class_args["dim"] = encoder_dim
        self.pool = pool_class(**pool_class_args)
        self.pooling = pooling
        if "WPCA" in img_model_name:
            pool_dim = encoder_dim * self.pool.num_clusters
            self.num_pcs = int(list(filter(lambda x: "WPCA" in x, img_model_name.split("_")))[
                               0].replace('WPCA', ''))
            pca_conv = nn.Conv2d(pool_dim, self.num_pcs,
                                 kernel_size=(1, 1), stride=1, padding=0)
            pcalayer = nn.Sequential(*[pca_conv, Flatten(), L2Norm(dim=-1)])
            self.WPCA = pcalayer
        else:
            self.WPCA = None

    def forward(self, x):
        if "dinov2" in self.arch:
            x = self.encoder(x, is_training=True)
        else:
            x = self.encoder(x)
        if any(arch in self.pooling for arch in ['applar', 'shadowvladscam']):
            heatmap, output = self.pool(x)
            heatmap, output = heatmap[0].data.cpu().numpy().sum(
                0), output
        else:
            output = self.pool(x)
            heatmap = np.random.rand(30, 40)


        if self.WPCA is not None:
            output = self.WPCA(output.unsqueeze(-1).unsqueeze(-1))

        return heatmap, output