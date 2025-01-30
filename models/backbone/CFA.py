import torch.nn as nn

class CrossFusionAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0., proj_drop=0.):
        super(CrossFusionAttention, self).__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.proj_q = nn.Linear(dim, dim, bias=False)
        self.proj_k = nn.Linear(dim, dim, bias=False)
        self.proj_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2, mask=None):
        B, L1, C = x1.shape
        _, L2, _ = x2.shape

        # [B, H, L, C]
        q = self.proj_q(x1).view(B, L1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.proj_k(x2).view(B, L2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.proj_v(x2).view(B, L2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'), )

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, L1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x