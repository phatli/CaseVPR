import torch.nn as nn

from .nn_module import NetVLAD
from .seqnet import seqNet
from .backbone import load_basic_model, get_pure_dinov2
from .nn_module import L2Norm
from .nn_module import Flatten as Flatten_img
from .cricavpr.backbone.vision_transformer import vit_base
from .casenet import CaseNet

class HVPR_SeqNet(nn.Module):
    def __init__(self, w = 3, arch = "vgg16", output_dim = 4096, seq_len = 5):
        super().__init__()
        self.encoder, encoder_dim = load_basic_model(arch)
        self.pool = NetVLAD(dim=encoder_dim)
        pool_dim = encoder_dim * self.pool.num_clusters
        pca_conv = nn.Conv2d(pool_dim, output_dim,
                                kernel_size=(1, 1), stride=1, padding=0)
        self.WPCA = nn.Sequential(*[pca_conv, Flatten_img(), L2Norm(dim=-1)])
        self.seq_encoder = seqNet(output_dim, output_dim, 5, w=w)
        self.seq_length = seq_len

    def forward(self, x, single_img=False):
        img_features = self.encoder(x)
        img_desc = self.pool(img_features)
        img_desc = self.WPCA(img_desc.unsqueeze(-1).unsqueeze(-1))

        if single_img:
            seq_desc = None
        else:
            seq_desc = self.seq_encoder(img_desc.unsqueeze(0))
        
        return img_desc, seq_desc

class HVPR_CaseNet(nn.Module):
    def __init__(self, seq_len=5, encoder_type="bs_d_c", is_pure=False):
        super().__init__()
        if is_pure:
            self.encoder = get_pure_dinov2(num_unfrozen_blocks=2)
        else:
            self.encoder = vit_base(patch_size=14, img_size=518,
                             init_values=1, block_chunks=0)
        self.pool = CaseNet(seq_len=seq_len, encode_type=encoder_type)
        self.seq_length = seq_len

    def forward(self, x, single_img=False):
        features = self.encoder(x, is_training=True)
        img_desc, seq_desc = self.pool(features, output_img=True, single_img = single_img)
        return img_desc, seq_desc
