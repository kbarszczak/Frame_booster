import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch


"""
This module contains fbnet v6_4 with the following features:
- Overall Attention U-Net with single feature warping
- Warp only one image
- Dense CNN flow prediciton architecture
- Flow concatenation not addition
- Attention gate usage
- Encoder with filters outputs on each level but included feature stacking on the input of the conv2dblock (each block inputs is a stack of every previous layers outputs properly resized by avgpooling2d)
- Decoder features addition
- Use of upsample and resize layers to change size
- Use of the proper coef to the flow upscaling
- Use of custom Conv2dBlock (in_channels -> out_channels -> out_channels, more conv2d layers at each level, 2 vs 1)
- Last level of the decoder is processed by Conv2dBlock
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
        
        # flow estimation layers
        input_channels = 2*input_channels + (2 if level != 4 else 0)
        self.cnn_1 = nn.Conv2d(input_channels, fcounts[0], fsizes[0], padding=fpads[0])
        self.cnn_2 = nn.Conv2d(input_channels + fcounts[0], fcounts[1], fsizes[1], padding=fpads[1])
        self.cnn_3 = nn.Conv2d(input_channels + fcounts[0] + fcounts[1], fcounts[2], fsizes[2], padding=fpads[2])
        self.cnn_4 = nn.Conv2d(input_channels + fcounts[0] + fcounts[1] + fcounts[2], fcounts[3], fsizes[3], padding=fpads[3])
        self.cnn_5 = nn.Conv2d(input_channels + fcounts[0] + fcounts[1] + fcounts[2] + fcounts[3], fcounts[4], fsizes[4], padding=fpads[4])
        
        self.cnn_out = nn.Conv2d(input_channels + fcounts[0] + fcounts[1] + fcounts[2] + fcounts[3] + fcounts[4], 2, 3, 1, 1)
        self.upsample = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)
        
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
        flow_upsampled = self.upsample(flow) * 2  # we multiply by 2 because of the upsampling the scale
        
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
        self.upsample = nn.Upsample(scale_factor=(2, 2), mode='bilinear')
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
    

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(Conv2dBlock, self).__init__(**kwargs)
        
        self.cnn_1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.act_1 = nn.PReLU()
        self.cnn_2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.act_2 = nn.PReLU()

    def forward(self, x):
        x = self.act_1(self.cnn_1(x))
        x = self.act_2(self.cnn_2(x))
        return x
    

class FBNet(nn.Module):
    def __init__(self, input_shape, device, filters=[64, 64, 64, 64], **kwargs):
        super(FBNet, self).__init__(**kwargs)
        
        self.b = input_shape[0]
        self.c = input_shape[1]
        self.h = input_shape[2]
        self.w = input_shape[3]
        
        # ------------- Feature encoding layers
        self.cnn_block_r1 = Conv2dBlock(self.c, filters[0], 3, 1, 1)
        self.cnn_block_r2 = Conv2dBlock(self.c + filters[0], filters[1], 3, 1, 1)
        self.cnn_block_r3 = Conv2dBlock(self.c + filters[0] + filters[1], filters[2], 3, 1, 1)
        self.cnn_block_r4 = Conv2dBlock(self.c + filters[0] + filters[1] + filters[2], filters[3], 3, 1, 1)
        
        self.downsample_1_2 = torchvision.transforms.Resize((self.h//2, self.w//2), antialias=True)
        self.downsample_1_3 = torchvision.transforms.Resize((self.h//4, self.w//4), antialias=True)
        self.downsample_1_4 = torchvision.transforms.Resize((self.h//8, self.w//8), antialias=True)
        
        self.avg_r1_1 = nn.AvgPool2d((2, 2))
        self.avg_r1_2 = nn.AvgPool2d((4, 4))
        self.avg_r1_3 = nn.AvgPool2d((8, 8))
        self.avg_r2_1 = nn.AvgPool2d((2, 2))
        self.avg_r2_2 = nn.AvgPool2d((4, 4))
        self.avg_r3_1 = nn.AvgPool2d((2, 2))
        
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
            g_filters=filters[0], 
            x_filters=filters[0], 
            filters=filters[0]
        )
        self.attention_r2 = AttentionGate(
            g_filters=filters[1], 
            x_filters=filters[1], 
            filters=filters[1]
        )
        self.attention_r3 = AttentionGate(
            g_filters=filters[2], 
            x_filters=filters[2], 
            filters=filters[2]
        )
        
        # ------------- Feature decoding layers
        self.conv_dec_block_r4 = Conv2dBlock(filters[3], filters[2], 3, 1, 1)
        self.up_r4 = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)
        
        self.conv_dec_block_r3 = Conv2dBlock(filters[2], filters[1], 3, 1, 1)
        self.up_r3 = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)
        
        self.conv_dec_block_r2 = Conv2dBlock(filters[1], filters[0], 3, 1, 1)
        self.up_r2 = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)
        
        self.conv_dec_block_r1 = Conv2dBlock(filters[0], filters[0], 3, 1, 1)
    
        self.cnn_out = nn.Conv2d(filters[0], 3, 3, 1, 1)
        self.act_out = TReLU()
        
    def forward(self, left, right):
        # ------------- Process left input
        # downsample input
        left_2 = self.downsample_1_2(left)
        left_3 = self.downsample_1_3(left)
        left_4 = self.downsample_1_4(left)
        
        # first conv block
        input_left_cnn_r1 = self.cnn_block_r1(left)
        intut_left_avg_r1_1 = self.avg_r1_1(input_left_cnn_r1)
        intut_left_avg_r1_2 = self.avg_r1_2(input_left_cnn_r1)
        intut_left_avg_r1_3 = self.avg_r1_3(input_left_cnn_r1)
        
        # second conv block
        input_left_cnn_r2 = self.cnn_block_r2(torch.cat([left_2, intut_left_avg_r1_1], dim=1))
        intut_left_avg_r2_1 = self.avg_r2_1(input_left_cnn_r2)
        intut_left_avg_r2_2 = self.avg_r2_2(input_left_cnn_r2)
        
        # third conv block
        input_left_cnn_r3 = self.cnn_block_r3(torch.cat([left_3, intut_left_avg_r1_2, intut_left_avg_r2_1], dim=1))
        intut_left_avg_r3_1 = self.avg_r3_1(input_left_cnn_r3)
        
        # fourth conv block
        input_left_cnn_r4 = self.cnn_block_r4(torch.cat([left_4, intut_left_avg_r1_3, intut_left_avg_r2_2, intut_left_avg_r3_1], dim=1))
        
        # output:
        # * input_left_cnn_r1
        # * input_left_cnn_r2
        # * input_left_cnn_r3
        # * input_left_cnn_r4

        # ------------- Process right input
        # downsample input
        right_2 = self.downsample_1_2(right)
        right_3 = self.downsample_1_3(right)
        right_4 = self.downsample_1_4(right)
        
        # first conv block
        input_right_cnn_r1 = self.cnn_block_r1(right)
        intut_right_avg_r1_1 = self.avg_r1_1(input_right_cnn_r1)
        intut_right_avg_r1_2 = self.avg_r1_2(input_right_cnn_r1)
        intut_right_avg_r1_3 = self.avg_r1_3(input_right_cnn_r1)
        
        # second conv block
        input_right_cnn_r2 = self.cnn_block_r2(torch.cat([right_2, intut_right_avg_r1_1], dim=1))
        intut_right_avg_r2_1 = self.avg_r2_1(input_right_cnn_r2)
        intut_right_avg_r2_2 = self.avg_r2_2(input_right_cnn_r2)
        
        # third conv block
        input_right_cnn_r3 = self.cnn_block_r3(torch.cat([right_3, intut_right_avg_r1_2, intut_right_avg_r2_1], dim=1))
        intut_right_avg_r3_1 = self.avg_r3_1(input_right_cnn_r3)
        
        # fourth conv block
        input_right_cnn_r4 = self.cnn_block_r4(torch.cat([right_4, intut_right_avg_r1_3, intut_right_avg_r2_2, intut_right_avg_r3_1], dim=1))
        
        # output:
        # * input_right_cnn_r1
        # * input_right_cnn_r2
        # * input_right_cnn_r3
        # * input_right_cnn_r4
        
        # ------------- Warp features
        warp_r4, flow_r4 = self.flow_warp_r4(input_left_cnn_r4, input_right_cnn_r4, None)
        warp_r3, flow_r3 = self.flow_warp_r3(input_left_cnn_r3, input_right_cnn_r3, flow_r4)
        warp_r2, flow_r2 = self.flow_warp_r2(input_left_cnn_r2, input_right_cnn_r2, flow_r3)
        warp_r1,  _ = self.flow_warp_r1(input_left_cnn_r1, input_right_cnn_r1, flow_r2)
        
        # output:
        # * warp_r4
        # * warp_r3
        # * warp_r2
        # * warp_r1
        
        # ------------- Decode features with attention 
        # row 4
        warp_r4 = self.conv_dec_block_r4(warp_r4)
        warp_r4_up = self.up_r4(warp_r4)
        
        # row 3
        warp_r3 = self.attention_r3(warp_r4, warp_r3)
        warp_r3 = warp_r3 + warp_r4_up
        warp_r3 = self.conv_dec_block_r3(warp_r3)
        warp_r3_up = self.up_r3(warp_r3)
        
        # row 2
        warp_r2 = self.attention_r2(warp_r3, warp_r2)
        warp_r2 = warp_r2 + warp_r3_up
        warp_r2 = self.conv_dec_block_r2(warp_r2)
        warp_r2_up = self.up_r2(warp_r2)
        
        # row 1
        warp_r1 = self.attention_r1(warp_r2, warp_r1)
        warp_r1 = warp_r1 + warp_r2_up
        warp_r1 = self.conv_dec_block_r1(warp_r1)
        result = self.act_out(self.cnn_out(warp_r1))
        
        # output:
        # * result
        
        return result
