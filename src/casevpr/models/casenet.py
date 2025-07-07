import torch
from torch import nn
import torch.nn.functional
import math
import einops
from .nn_module import Flatten, L2Norm, GeM

class CaseNet(nn.Module):
    def __init__(self, seq_len=5, encode_type="b_sd_c"):
        super().__init__()
        self.aggregation = nn.Sequential(L2Norm(), GeM(work_with_tokens=None), Flatten())

        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=16, dim_feedforward=2048, activation="gelu", dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2) # Cross-image encoder
        self.seq_len = seq_len
        self.encode_type = encode_type

    def forward(self, x, output_img=False, single_img=False):
        B,P,D = x["x_prenorm"].shape
        W = H = int(math.sqrt(P-1))
        x0 = x["x_norm_clstoken"]
        x_p = x["x_norm_patchtokens"].view(B,W,H,D).permute(0, 3, 1, 2) 

        x10,x11,x12,x13 = self.aggregation(x_p[:,:,0:8,0:8]),self.aggregation(x_p[:,:,0:8,8:]),self.aggregation(x_p[:,:,8:,0:8]),self.aggregation(x_p[:,:,8:,8:])
        x20,x21,x22,x23,x24,x25,x26,x27,x28 = self.aggregation(x_p[:,:,0:5,0:5]),self.aggregation(x_p[:,:,0:5,5:11]),self.aggregation(x_p[:,:,0:5,11:]),\
                                        self.aggregation(x_p[:,:,5:11,0:5]),self.aggregation(x_p[:,:,5:11,5:11]),self.aggregation(x_p[:,:,5:11,11:]),\
                                        self.aggregation(x_p[:,:,11:,0:5]),self.aggregation(x_p[:,:,11:,5:11]),self.aggregation(x_p[:,:,11:,11:])
        x = [i.unsqueeze(1) for i in [x0,x10,x11,x12,x13,x20,x21,x22,x23,x24,x25,x26,x27,x28]]

        x = torch.cat(x,dim=1)
        if self.encode_type == "b_sd_c" and not single_img:
            x = einops.rearrange(
                x, '(b s) d c -> b (s d) c', s=self.seq_len, d=14)
        if self.encode_type == "sd_b_c" and not single_img:
            x = einops.rearrange(
                x, '(b s) d c -> (s d) b c', s=self.seq_len, d=14)
        x = self.encoder(x) # requires (seq_len, B, D)
        if self.encode_type == "b_sd_c" and not single_img:
            x = einops.rearrange(
                x, 'b (s d) c -> b s (d c)', s=self.seq_len, d=14)
        elif self.encode_type == "bs_d_c" and not single_img:
            x = einops.rearrange(
                x, '(b s) d c -> b s (d c)', s=self.seq_len, d=14)
        elif self.encode_type == "sd_b_c" and not single_img:
            x = einops.rearrange(
                x, '(s d) b c -> b s (d c)', s=self.seq_len, d=14)
        elif single_img:
            x = x.view(B, 14*D)
            x = torch.nn.functional.normalize(x, p=2, dim=-1)
            if output_img:
                return x, None
            return None
        if output_img:
            assert x.shape[0] == 1, "Only one image can be outputted, therefore batch size must be 1"
            img_desc = x.squeeze(0)
            img_desc = torch.nn.functional.normalize(img_desc, p=2, dim=-1)

        x = x.mean(dim=1).squeeze(1)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        if output_img:
            return img_desc, x
        return x