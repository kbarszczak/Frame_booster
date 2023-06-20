import torch.nn.functional as F
import torch.nn as nn
import torch


"""
This module contains fbnet v6_2 with the following features:
- Overall Attention U-Net with single feature warping
- Warp only one image
- Dense CNN flow prediciton architecture
- Flow concatenation not addition
- Attention gate usage
- Encoder with filters outputs on each level (without feature stacking)
- Decoder features concatenation
- Use of conv2d transpose to change size
- Use of the proper coef to the flow upscaling
"""


class TReLU(nn.Module):
    def __init__(self, lower=0.0, upper=1.0, **kwargs):
        super(TReLU, self).__init__(**kwargs)
        self.lower = lower
        self.upper = upper

    def forward(self, x):
        return torch.clip(x, min=self.lower, max=self.upper)


class FlowFeatureWarp(nn.Module):
    def __init__(self, input_channels, level,
                 flow_info = {
                    "filter_counts": [128, 128, 96, 64, 32],
                    "filter_sizes": [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)],
                    "filter_paddings": [1, 1, 1, 1, 1]
                 }, **kwargs):
        super(FlowFeatureWarp, self).__init__(**kwargs)
        
        fcounts = flow_info['filter_counts']
        fsizes = flow_info['filter_sizes']
        fpads = flow_info['filter_paddings']
        
        assert len(fcounts) == len(fsizes) == len(fpads) == 5, "Filter options should have the same size of 5 elements"
        
        # cost volume layers
        # todo: implement in the future
        
        # flow estimation layers
        input_channels = 2*input_channels + (2 if level != 4 else 0)
        self.cnn_1 = nn.Conv2d(input_channels, fcounts[0], fsizes[0], padding=fpads[0])
        self.cnn_2 = nn.Conv2d(input_channels + fcounts[0], fcounts[1], fsizes[1], padding=fpads[1])
        self.cnn_3 = nn.Conv2d(input_channels + fcounts[0] + fcounts[1], fcounts[2], fsizes[2], padding=fpads[2])
        self.cnn_4 = nn.Conv2d(input_channels + fcounts[0] + fcounts[1] + fcounts[2], fcounts[3], fsizes[3], padding=fpads[3])
        self.cnn_5 = nn.Conv2d(input_channels + fcounts[0] + fcounts[1] + fcounts[2] + fcounts[3], fcounts[4], fsizes[4], padding=fpads[4])
        
        self.cnn_out = nn.Conv2d(input_channels + fcounts[0] + fcounts[1] + fcounts[2] + fcounts[3] + fcounts[4], 2, 3, 1, 1)
        self.cnn_upsample = nn.ConvTranspose2d(2, 2, 4, stride=2, padding=1)
        
        # context layers
        # todo: implement context layer that upgrades the flow at each level
        
        self.level = level

    def forward(self, input_1, input_2, upsampled_flow):
        if self.level != 4:
            x = FlowFeatureWarp.warp(input_2, upsampled_flow)
            x = torch.cat([input_1, x, upsampled_flow], dim=1)
        else:
            x = torch.cat([input_1, input_2], dim=1)
        
        x = torch.cat([self.cnn_1(x), x], dim=1)
        x = torch.cat([self.cnn_2(x), x], dim=1)
        x = torch.cat([self.cnn_3(x), x], dim=1)
        x = torch.cat([self.cnn_4(x), x], dim=1)
        x = torch.cat([self.cnn_5(x), x], dim=1)
        flow = self.cnn_out(x)
        
        input_2_warped = FlowFeatureWarp.warp(input_2, flow * 0.5)
        flow_upsampled = self.cnn_upsample(flow) * 2  # we multiply by 2 because of the upsampling the scale
        
        return input_2_warped, flow_upsampled
    
    @staticmethod
    def warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image.size()

        xx = torch.arange(0, W).view(1 ,-1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1 ,1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)

        grid = torch.cat((xx, yy), 1).float()
        if image.is_cuda:
            grid = grid.cuda()

        vgrid = grid + flow
        vgrid[:, 0, :, :] = 2.0 * vgrid[: ,0 ,: ,:].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[: ,1 ,: ,:].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        flow = flow.permute(0, 2, 3, 1)
        output = F.grid_sample(image, vgrid, align_corners=False)

        return output
    

class AttentionGate(nn.Module):
    def __init__(self, g_filters, x_filters, filters, **kwargs):
        super(AttentionGate, self).__init__(**kwargs)
        
        self.gcnn = nn.Conv2d(g_filters, filters, 3, 1, 1)
        self.xcnn = nn.Conv2d(x_filters, filters, 3, 2, 1)
        self.ocnn = nn.Conv2d(filters, 1, 1)
        self.upsample = nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1)
        self.act = nn.PReLU()
        self.out_act = nn.Sigmoid()

    def forward(self, g, x):
        xcnn = self.xcnn(x)
        gcnn = self.gcnn(g)
        
        xg = xcnn + gcnn
        
        xg = self.act(xg)
        xg = self.ocnn(xg)
        xg = self.out_act(xg)
        xg = self.upsample(xg)
        
        return x * xg


class FBNet(nn.Module):
    def __init__(self, input_shape, device, filters=[32, 48, 64, 64], **kwargs):
        super(FBNet, self).__init__(**kwargs)
        
        self.b = input_shape[0]
        self.c = input_shape[1]
        self.h = input_shape[2]
        self.w = input_shape[3]
        
        # ------------- Feature encoding layers
        self.cnn_r1_1 = nn.Conv2d(self.c, filters[0], 3, 1, 1)
        self.cnn_r1_2 = nn.Conv2d(filters[0], filters[0], 3, 2, 1)
        self.cnn_r2_1 = nn.Conv2d(filters[0], filters[1], 3, 1, 1)
        self.cnn_r2_2 = nn.Conv2d(filters[1], filters[1], 3, 2, 1)
        self.cnn_r3_1 = nn.Conv2d(filters[1], filters[2], 3, 1, 1)
        self.cnn_r3_2 = nn.Conv2d(filters[2], filters[2], 3, 2, 1)
        self.cnn_r4_1 = nn.Conv2d(filters[2], filters[3], 3, 1, 1)
        
        self.act_r1_1 = nn.PReLU()
        self.act_r1_2 = nn.PReLU()
        self.act_r2_1 = nn.PReLU()
        self.act_r2_2 = nn.PReLU()
        self.act_r3_1 = nn.PReLU()
        self.act_r3_2 = nn.PReLU()
        self.act_r4_1 = nn.PReLU()
        
        # ------------- Feature warping layers 
        self.flow_warp_r1 = FlowFeatureWarp(
            input_channels=filters[0],
            level = 1
        )
        self.flow_warp_r2 = FlowFeatureWarp(
            input_channels=filters[1],
            level = 2 
        )
        self.flow_warp_r3 = FlowFeatureWarp(
            input_channels=filters[2],
            level = 3 
        )
        self.flow_warp_r4 = FlowFeatureWarp(
            input_channels=filters[3],
            level = 4 
        )
        
        # ------------- Attention layers
        self.attention_r1 = AttentionGate(
            g_filters=filters[1], 
            x_filters=filters[0], 
            filters=filters[0]
        )
        self.attention_r2 = AttentionGate(
            g_filters=filters[2], 
            x_filters=filters[1], 
            filters=filters[1]
        )
        self.attention_r3 = AttentionGate(
            g_filters=filters[3], 
            x_filters=filters[2], 
            filters=filters[2]
        )
        
        # ------------- Feature decoding layers
        self.cnn_r4_2 = nn.ConvTranspose2d(filters[3], filters[2], 4, 2, 1)
        self.cnn_r3_3 = nn.Conv2d(2*filters[2], filters[2], 3, 1, 1)
        self.cnn_r3_4 = nn.ConvTranspose2d(filters[2], filters[1], 4, 2, 1)
        self.cnn_r2_3 = nn.Conv2d(2*filters[1], filters[1], 3, 1, 1)
        self.cnn_r2_4 = nn.ConvTranspose2d(filters[1], filters[0], 4, 2, 1)
        self.cnn_r1_3 = nn.Conv2d(2*filters[0], filters[0], 3, 1, 1)
        self.cnn_r1_4 = nn.Conv2d(filters[0], 3, 3, 1, 1)
        
        self.act_r3_3 = nn.PReLU()
        self.act_r2_3 = nn.PReLU()
        self.act_r1_3 = nn.PReLU()
        self.act_out = TReLU()
        
    def forward(self, left, right):
        # ------------- Process left input
        input_left_cnn_r1_1 = self.act_r1_1(self.cnn_r1_1(left))
        input_left_cnn_r1_2 = self.act_r1_2(self.cnn_r1_2(input_left_cnn_r1_1))
        
        input_left_cnn_r2_1 = self.act_r2_1(self.cnn_r2_1(input_left_cnn_r1_2))
        input_left_cnn_r2_2 = self.act_r2_2(self.cnn_r2_2(input_left_cnn_r2_1))
        
        input_left_cnn_r3_1 = self.act_r3_1(self.cnn_r3_1(input_left_cnn_r2_2))
        input_left_cnn_r3_2 = self.act_r3_2(self.cnn_r3_2(input_left_cnn_r3_1))
        
        input_left_cnn_r4_1 = self.act_r4_1(self.cnn_r4_1(input_left_cnn_r3_2))
        
        # output:
        # * input_left_cnn_r1_1
        # * input_left_cnn_r2_1
        # * input_left_cnn_r3_1
        # * input_left_cnn_r4_1
        
        # ------------- Process right input
        input_right_cnn_r1_1 = self.act_r1_1(self.cnn_r1_1(right))
        input_right_cnn_r1_2 = self.act_r1_2(self.cnn_r1_2(input_right_cnn_r1_1))
        
        input_right_cnn_r2_1 = self.act_r2_1(self.cnn_r2_1(input_right_cnn_r1_2))
        input_right_cnn_r2_2 = self.act_r2_2(self.cnn_r2_2(input_right_cnn_r2_1))
        
        input_right_cnn_r3_1 = self.act_r3_1(self.cnn_r3_1(input_right_cnn_r2_2))
        input_right_cnn_r3_2 = self.act_r3_2(self.cnn_r3_2(input_right_cnn_r3_1))
        
        input_right_cnn_r4_1 = self.act_r4_1(self.cnn_r4_1(input_right_cnn_r3_2))
        
        # output:
        # * input_right_cnn_r1_1
        # * input_right_cnn_r2_1
        # * input_right_cnn_r3_1
        # * input_right_cnn_r4_1
        
        # ------------- Warp features
        warp_r4, flow_r4 = self.flow_warp_r4(input_right_cnn_r4_1, input_left_cnn_r4_1, None)
        warp_r3, flow_r3 = self.flow_warp_r3(input_right_cnn_r3_1, input_left_cnn_r3_1, flow_r4)
        warp_r2, flow_r2 = self.flow_warp_r2(input_right_cnn_r2_1, input_left_cnn_r2_1, flow_r3)
        warp_r1, flow_r1 = self.flow_warp_r1(input_right_cnn_r1_1, input_left_cnn_r1_1, flow_r2)
        
        # output:
        # * warp_r4
        # * warp_r3
        # * warp_r2
        # * warp_r1
        
        # ------------- Decode features with attention 
        # row 4
        input_cnn_r4_2 = self.cnn_r4_2(warp_r4)
        intput_attention_r3 = self.attention_r3(warp_r4, warp_r3)
        intput_cat_r3 = torch.cat([intput_attention_r3, input_cnn_r4_2], dim=1)
        input_cnn_r3_3 = self.act_r3_3(self.cnn_r3_3(intput_cat_r3))
        
        # row 3
        input_cnn_r3_4 = self.cnn_r3_4(input_cnn_r3_3)
        intput_attention_r2 = self.attention_r2(input_cnn_r3_3, warp_r2)
        intput_cat_r2 = torch.cat([intput_attention_r2, input_cnn_r3_4], dim=1)
        input_cnn_r2_3 = self.act_r2_3(self.cnn_r2_3(intput_cat_r2))
        
        # row 2
        input_cnn_r2_4 = self.cnn_r2_4(input_cnn_r2_3)
        intput_attention_r1 = self.attention_r1(input_cnn_r2_3, warp_r1)
        intput_cat_r1 = torch.cat([intput_attention_r1, input_cnn_r2_4], dim=1)
        input_cnn_r1_3 = self.act_r1_3(self.cnn_r1_3(intput_cat_r1))
        
        # row 1
        input_cnn_r1_4 = self.act_out(self.cnn_r1_4(input_cnn_r1_3))
        
        return input_cnn_r1_4
