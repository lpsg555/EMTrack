import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

from .attn import Attention
from ..backbone.CFA import CrossFusionAttention

class CEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1, embed_dim=768):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.keep_ratio_search = keep_ratio_search
        self.cross_attn = CrossFusionAttention(dim=embed_dim, num_heads=8)

    def forward(self, x1, z, global_index_template, global_index_search, mask=None, ce_template_mask=None, keep_ratio_search=None,
                add_cls_token=False, query_len=1):

        x2 = x1 + self.cross_attn(x1, z)
        x2 = x2 + self.drop_path(self.mlp1(self.norm1(x2)))  # cross_attn + FFN

        x_attn, attn = self.attn(self.norm2(x2), mask, True)  # self_attn + FFN
        x = x2 + self.drop_path(x_attn)

        lens_t = global_index_template.shape[1]

        removed_index_search = None
        x = x + self.drop_path(self.mlp2(self.norm3(x)))
        return x, z, global_index_template, global_index_search, removed_index_search, attn
