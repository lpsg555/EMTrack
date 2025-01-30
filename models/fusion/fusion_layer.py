import torch
import torch.nn as nn
from models.fusion.CBAM import ChannelAttention,CBAM


def Conv_Block(input_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(input_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(inplace=True),
    )


class Fusion_Block(nn.Module):
    def __init__(self, dim=32):
        super(Fusion_Block, self).__init__()

        self.conv_img = Conv_Block(3, dim * 4)
        self.conv_evt = Conv_Block(3, dim)
        self.conv_fusion = Conv_Block(dim * 5, dim * 2)
        self.conv_img2 = Conv_Block(dim * 4, dim)
        self.conv_evt2 = Conv_Block(dim ,dim)

        self.adaptPool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)

        self.CA_1 = ChannelAttention(dim * 5)
        self.CA_2 = ChannelAttention(dim * 2)

        self.conv_output = Conv_Block(dim * 2, dim * 2)

    def forward(self, img, evt):
        img1 = self.conv_img(img)
        evt1 = self.conv_evt(evt)
        img2 = self.conv_img2(img1)
        evt2 = self.conv_evt2(evt1)


        x = torch.cat([img1, evt1], dim=1)
        x = self.CA_1(x)
        x = self.conv_fusion(x)
        x = self.adaptPool(x)



        low_att, high_att = torch.chunk(x, 2, dim=1)
        img_attn = low_att * img2
        evt_attn = low_att * evt2


        feat_fusion = torch.cat([img_attn,evt_attn],dim=1)

        feat_fusion = self.CA_2(feat_fusion)
        feat_fusion = self.conv_output(feat_fusion)

        return feat_fusion