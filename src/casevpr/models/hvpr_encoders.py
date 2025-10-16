import torch.nn as nn
from typing import List, Optional

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
        self.meta = {"outputdim": output_dim}

    def forward(self, x, single_img=False):
        img_features = self.encoder(x)
        img_desc = self.pool(img_features)
        img_desc = self.WPCA(img_desc.unsqueeze(-1).unsqueeze(-1))

        if single_img:
            seq_desc = None
        else:
            total_frames = img_desc.shape[0]
            if total_frames % self.seq_length != 0:
                raise ValueError(
                    f"Input contains {total_frames} frames which is not a multiple of seq_len={self.seq_length}"
                )
            batch_size = total_frames // self.seq_length
            seq_input = img_desc.view(batch_size, self.seq_length, -1)
            seq_desc = self.seq_encoder(seq_input)
        
        return img_desc, seq_desc

class HVPR_CaseNet(nn.Module):
    def __init__(
        self,
        seq_len: int = 5,
        encoder_type: str = "bs_d_c",
        is_pure: bool = False,
        adapters_only: bool = True,
        trainable_block_count: int = 1,
        is_training: bool = True,
    ) -> None:
        super().__init__()
        self.seq_length = seq_len
        if is_pure:
            self.encoder = get_pure_dinov2(num_unfrozen_blocks=2)
        else:
            self.encoder = vit_base(patch_size=14, img_size=518, init_values=1, block_chunks=0)
            if adapters_only:
                self._freeze_encoder_adapters(trainable_block_count)
        self.pooling = CaseNet(seq_len=seq_len, encode_type=encoder_type)
        self.is_training = is_training

    def _freeze_encoder_adapters(self, trainable_block_count: int) -> None:
        for _, param in self.encoder.named_parameters():
            param.requires_grad = False

        blocks = self._collect_transformer_blocks()
        if not blocks:
            return

        trainable_block_count = max(1, min(trainable_block_count, len(blocks)))
        threshold_index = len(blocks) - trainable_block_count

        for block_idx, block in enumerate(blocks):
            if block_idx < threshold_index:
                continue
            for name, param in block.named_parameters():
                if "adapter" in name:
                    param.requires_grad = True

    def _collect_transformer_blocks(self) -> List[nn.Module]:
        if not hasattr(self.encoder, "blocks"):
            return []
        blocks_attr = self.encoder.blocks
        blocks: list[nn.Module] = []
        if isinstance(blocks_attr, nn.ModuleList):
            for item in blocks_attr:
                if isinstance(item, nn.ModuleList):
                    blocks.extend([module for module in item if not isinstance(module, nn.Identity)])
                else:
                    blocks.append(item)
        return blocks

    def forward(self, x, single_img=False):
        features = self.encoder(x, is_training=self.is_training)
        return_img = not self.training or single_img
        outputs = self.pooling(features, output_img=return_img, single_img=single_img)
        if return_img:
            img_desc, seq_desc = outputs
            return img_desc, seq_desc
        return outputs

    @property
    def pool(self):
        return self.pooling
