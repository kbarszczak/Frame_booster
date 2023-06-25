import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch


"""
This module contains fbnet v6_7 with the following features:
- TReLU output activation
- Overall U-Net with double feature warping
- Warp both images
- Dense CNN flow prediciton architecture
- Flow addition
- Encoder with passing level information 2 layers below (see FILM implementation)
- Decoder features addition
- Use of upsample and resize layers to change size
- Use of the proper coef to the flow upscaling
- Use of custom Conv2dBlock (in_channels -> out_channels -> out_channels, more conv2d layers at each level, 2 vs 1)
- Last level of the decoder is processed by Conv2dBlock
- Use of flow attention layer
"""


class TReLU(nn.Module):
    def __init__(self, lower=0.0, upper=1.0, **kwargs):
        super(TReLU, self).__init__(**kwargs)
        self.lower = lower
        self.upper = upper

    def forward(self, x):
        return torch.clip(x, min=self.lower, max=self.upper)
    

class AttentionGate(nn.Module):
    def __init__(self, g_filters, x_filters, filters, **kwargs):
        super(AttentionGate, self).__init__(**kwargs)
        
        self.gcnn = nn.Conv2d(g_filters, filters, 3, 1, 1)
        self.xcnn = nn.Conv2d(x_filters, filters, 3, 1, 1)
        self.ocnn = nn.Conv2d(filters, 1, 1)
        self.act = nn.LeakyReLU(0.1)
        self.out_act = nn.Sigmoid()
        self.upsample = None  # for compability

    def forward(self, g, x):
        xcnn = self.xcnn(x)
        gcnn = self.gcnn(g)

        xg = xcnn + gcnn
        
        xg = self.act(xg)
        xg = self.ocnn(xg)
        xg = self.out_act(xg)

        return x * xg
    

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, padding_mode='zeros', **kwargs):
        super(Conv2dBlock, self).__init__(**kwargs)
        
        self.cnn_1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode)
        self.act_1 = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act_1(self.cnn_1(x))


class FlowFeatureWarp(nn.Module):
    def __init__(self, input_channels, level,
                 flow_info = {
                    "filter_counts": [32, 32, 32, 32, 32],
                    "filter_sizes": [(9, 9), (7, 7), (5, 5), (3, 3), (3, 3)],
                    "filter_paddings": [4, 3, 2, 1, 1]
                 }, **kwargs):
        super(FlowFeatureWarp, self).__init__(**kwargs)
        
        fcounts = flow_info['filter_counts']
        fsizes = flow_info['filter_sizes']
        fpads = flow_info['filter_paddings']
        
        assert len(fcounts) == len(fsizes) == len(fpads) == 5, "Filter options should have the same size of 5 elements"
        
        # flow estimation layers
        input_channels = 2*input_channels
        self.cnn_1 = Conv2dBlock(input_channels, fcounts[0], fsizes[0], 1, fpads[0], 'replicate')
        self.cnn_2 = Conv2dBlock(input_channels + fcounts[0], fcounts[1], fsizes[1], 1, fpads[1], 'replicate')
        self.cnn_3 = Conv2dBlock(input_channels + fcounts[0] + fcounts[1], fcounts[2], fsizes[2], 1, fpads[2], 'replicate')
        self.cnn_4 = Conv2dBlock(input_channels + fcounts[0] + fcounts[1] + fcounts[2], fcounts[3], fsizes[3], 1, fpads[3], 'replicate')
        self.cnn_5 = Conv2dBlock(input_channels + fcounts[0] + fcounts[1] + fcounts[2] + fcounts[3], fcounts[4], fsizes[4], 1, fpads[4], 'replicate')
        self.cnn_out = nn.Conv2d(input_channels + fcounts[0] + fcounts[1] + fcounts[2] + fcounts[3] + fcounts[4], 2, 1, 1, 0)
        self.upsample = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)
        
        self.level = level

    def forward(self, input_1, input_2, flow_1_2, flow_2_1):
        if self.level != 4:
            # flow from input 1 to input 2
            x_1_2 = FlowFeatureWarp.warp(input_1, flow_1_2)
            x_1_2 = torch.cat([x_1_2, input_2], dim=1)
            # flow from input 2 to input 1
            x_2_1 = FlowFeatureWarp.warp(input_2, flow_2_1)
            x_2_1 = torch.cat([input_1, x_2_1], dim=1)
        else:
            x_1_2 = torch.cat([input_1, input_2], dim=1)
            x_2_1 = torch.cat([input_1, input_2], dim=1)
            
        # flow correction from input 1 to input 2
        x_1_2 = torch.cat([self.cnn_1(x_1_2), x_1_2], dim=1)
        x_1_2 = torch.cat([self.cnn_2(x_1_2), x_1_2], dim=1)
        x_1_2 = torch.cat([self.cnn_3(x_1_2), x_1_2], dim=1)
        x_1_2 = torch.cat([self.cnn_4(x_1_2), x_1_2], dim=1)
        x_1_2 = torch.cat([self.cnn_5(x_1_2), x_1_2], dim=1)
        flow_change_1_2 = self.cnn_out(x_1_2)
        
        # flow correction from input 2 to input 1
        x_2_1 = torch.cat([self.cnn_1(x_2_1), x_2_1], dim=1)
        x_2_1 = torch.cat([self.cnn_2(x_2_1), x_2_1], dim=1)
        x_2_1 = torch.cat([self.cnn_3(x_2_1), x_2_1], dim=1)
        x_2_1 = torch.cat([self.cnn_4(x_2_1), x_2_1], dim=1)
        x_2_1 = torch.cat([self.cnn_5(x_2_1), x_2_1], dim=1)
        flow_change_2_1 = self.cnn_out(x_2_1)
        
        # add flow
        if self.level != 4:
            flow_1_2 = flow_1_2 + flow_change_1_2
            flow_2_1 = flow_2_1 + flow_change_2_1
        else:
            flow_1_2 = flow_change_1_2
            flow_2_1 = flow_change_2_1

        # warp features
        input_1_warped = FlowFeatureWarp.warp(input_1, flow_1_2 * 0.5)
        input_2_warped = FlowFeatureWarp.warp(input_2, flow_2_1 * 0.5)

        # upsample flow
        flow_1_2_upsampled = self.upsample(flow_1_2) * 2
        flow_2_1_upsampled = self.upsample(flow_2_1) * 2
            
        return input_1_warped, input_2_warped, flow_1_2_upsampled, flow_2_1_upsampled
    
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


class FBNet(nn.Module):
    def __init__(self, 
                input_shape, device,
                encoder_filters = [
                    [32, 32, 32, 32],  # encoder_filters_col_1
                    [32, 32, 32],  # encoder_filters_col_2
                    [32, 32]  # encoder_filters_col_3
                ], 
                decoder_filters = 32,
                **kwargs):
        super(FBNet, self).__init__(**kwargs)
        
        self.b = input_shape[0]
        self.c = input_shape[1]
        self.h = input_shape[2]
        self.w = input_shape[3]
        
        # ------------- Shared Conv2d, AvgPool2d & Resize encoding layers
        self.resize_1_2 = torchvision.transforms.Resize(size=(self.h//2, self.w//2), antialias=True)
        self.resize_1_4 = torchvision.transforms.Resize(size=(self.h//4, self.w//4), antialias=True)
        self.resize_1_8 = torchvision.transforms.Resize(size=(self.h//8, self.w//8), antialias=True)
        
        self.cnn_r1_c1 = Conv2dBlock(self.c, encoder_filters[0][0], 3, 1, 1)
        self.cnn_r2_c1 = Conv2dBlock(self.c, encoder_filters[0][1], 3, 1, 1)
        self.cnn_r3_c1 = Conv2dBlock(self.c, encoder_filters[0][2], 3, 1, 1)
        self.cnn_r4_c1 = Conv2dBlock(self.c, encoder_filters[0][3], 3, 1, 1)

        self.cnn_r2_c2 = Conv2dBlock(encoder_filters[0][0], encoder_filters[1][0], 3, 1, 1)
        self.cnn_r3_c2 = Conv2dBlock(encoder_filters[0][1], encoder_filters[1][1], 3, 1, 1)
        self.cnn_r4_c2 = Conv2dBlock(encoder_filters[0][2], encoder_filters[1][2], 3, 1, 1)

        self.cnn_r3_c3 = Conv2dBlock(encoder_filters[1][0], encoder_filters[2][0], 3, 1, 1)
        self.cnn_r4_c3 = Conv2dBlock(encoder_filters[1][1], encoder_filters[2][1], 3, 1, 1)
        
        self.avg_r2_c1 = nn.AvgPool2d(2)
        self.avg_r3_c1 = nn.AvgPool2d(2)
        self.avg_r4_c1 = nn.AvgPool2d(2)
        
        self.avg_r3_c2 = nn.AvgPool2d(2)
        self.avg_r4_c2 = nn.AvgPool2d(2)

        # ------------- Feature warping layers 
        self.flow_warp_r1 = FlowFeatureWarp(
            input_channels=encoder_filters[0][0],
            level = 1
        )
        self.flow_warp_r2 = FlowFeatureWarp(
            input_channels=encoder_filters[0][1] + encoder_filters[1][0],
            level = 2 
        )
        self.flow_warp_r3 = FlowFeatureWarp(
            input_channels=encoder_filters[0][2] + encoder_filters[1][1] + encoder_filters[2][0],
            level = 3 
        )
        self.flow_warp_r4 = FlowFeatureWarp(
            input_channels=encoder_filters[0][3] + encoder_filters[1][2] + encoder_filters[2][1],
            level = 4 
        )

        # ------------- Attention layers
        self.attention_r1 = AttentionGate(
            g_filters=4, 
            x_filters=encoder_filters[0][0], 
            filters=4
        )
        self.attention_r2 = AttentionGate(
            g_filters=4, 
            x_filters=encoder_filters[0][1] + encoder_filters[1][0], 
            filters=4
        )
        self.attention_r3 = AttentionGate(
            g_filters=4, 
            x_filters=encoder_filters[0][2] + encoder_filters[1][1] + encoder_filters[2][0], 
            filters=4
        )
        
        # ------------- Decoding Conv2d layers
        self.cnn_r4 = Conv2dBlock(encoder_filters[0][3] + encoder_filters[1][2] + encoder_filters[2][1], encoder_filters[0][2] + encoder_filters[1][1] + encoder_filters[2][0], 3, 1, 1)
        self.up_r4 = nn.Upsample(scale_factor=(2, 2), mode='bilinear')
        
        self.cnn_r3 = Conv2dBlock(encoder_filters[0][2] + encoder_filters[1][1] + encoder_filters[2][0], encoder_filters[0][1] + encoder_filters[1][0], 3, 1, 1)
        self.up_r3 = nn.Upsample(scale_factor=(2, 2), mode='bilinear')
        
        self.cnn_r2 = Conv2dBlock(encoder_filters[0][1] + encoder_filters[1][0], encoder_filters[0][0], 3, 1, 1)
        self.up_r2 = nn.Upsample(scale_factor=(2, 2), mode='bilinear')
        
        self.cnn_r1 = Conv2dBlock(encoder_filters[0][0], decoder_filters, 3, 1, 1)

        self.cnn_out = nn.Conv2d(decoder_filters, 3, 3, 1, 1)
        self.act_out = TReLU()

    def forward(self, input_1_left, input_1_right):
        # ------------- Process left input
        input_2_left = self.resize_1_2(input_1_left)
        input_3_left = self.resize_1_4(input_2_left)
        input_4_left = self.resize_1_8(input_3_left)
        
        # Feature extraction for layer 1
        input_1_left_cnn_r1_c1 = self.cnn_r1_c1(input_1_left)
        input_2_left_cnn_r2_c1 = self.cnn_r2_c1(input_2_left)
        input_3_left_cnn_r3_c1 = self.cnn_r3_c1(input_3_left)
        input_4_left_cnn_r4_c1 = self.cnn_r4_c1(input_4_left)

        # Downsample layer 1
        input_1_left_cnn_r2_c1 = self.avg_r2_c1(input_1_left_cnn_r1_c1)
        input_2_left_cnn_r3_c1 = self.avg_r3_c1(input_2_left_cnn_r2_c1)
        input_3_left_cnn_r4_c1 = self.avg_r4_c1(input_3_left_cnn_r3_c1)

        # Feature extraction for layer 2
        input_1_left_cnn_r2_c2 = self.cnn_r2_c2(input_1_left_cnn_r2_c1)
        input_2_left_cnn_r3_c2 = self.cnn_r3_c2(input_2_left_cnn_r3_c1)
        input_3_left_cnn_r4_c2 = self.cnn_r4_c2(input_3_left_cnn_r4_c1)

        # Downsample layer 2
        input_1_left_cnn_r3_c2 = self.avg_r3_c2(input_1_left_cnn_r2_c2)
        input_2_left_cnn_r4_c2 = self.avg_r4_c2(input_2_left_cnn_r3_c2)

        # Feature extraction for layer 3
        input_1_left_cnn_r3_c3 = self.cnn_r3_c3(input_1_left_cnn_r3_c2)
        input_2_left_cnn_r4_c3 = self.cnn_r4_c3(input_2_left_cnn_r4_c2)

        # Concatenate
        concat_left_row_2 = torch.cat([input_2_left_cnn_r2_c1, input_1_left_cnn_r2_c2], dim=1)
        concat_left_row_3 = torch.cat([input_3_left_cnn_r3_c1, input_2_left_cnn_r3_c2, input_1_left_cnn_r3_c3], dim=1)
        concat_left_row_4 = torch.cat([input_4_left_cnn_r4_c1, input_3_left_cnn_r4_c2, input_2_left_cnn_r4_c3], dim=1)
        
        # Feature extraction left side output: 
        # * input_1_left_cnn_r1_c1
        # * concat_left_row_2
        # * concat_left_row_3
        # * concat_left_row_4
        
        # ------------- Process right input
        input_2_right = self.resize_1_2(input_1_right)
        input_3_right = self.resize_1_4(input_2_right)
        input_4_right = self.resize_1_8(input_3_right)

        # Feature extraction for layer 1
        input_1_right_cnn_r1_c1 = self.cnn_r1_c1(input_1_right)
        input_2_right_cnn_r2_c1 = self.cnn_r2_c1(input_2_right)
        input_3_right_cnn_r3_c1 = self.cnn_r3_c1(input_3_right)
        input_4_right_cnn_r4_c1 = self.cnn_r4_c1(input_4_right)

        # Downsample layer 1
        input_1_right_cnn_r2_c1 = self.avg_r2_c1(input_1_right_cnn_r1_c1)
        input_2_right_cnn_r3_c1 = self.avg_r3_c1(input_2_right_cnn_r2_c1)
        input_3_right_cnn_r4_c1 = self.avg_r4_c1(input_3_right_cnn_r3_c1)

        # Feature extraction for layer 2
        input_1_right_cnn_r2_c2 = self.cnn_r2_c2(input_1_right_cnn_r2_c1)
        input_2_right_cnn_r3_c2 = self.cnn_r3_c2(input_2_right_cnn_r3_c1)
        input_3_right_cnn_r4_c2 = self.cnn_r4_c2(input_3_right_cnn_r4_c1)

        # Downsample layer 2
        input_1_right_cnn_r3_c2 = self.avg_r3_c2(input_1_right_cnn_r2_c2)
        input_2_right_cnn_r4_c2 = self.avg_r4_c2(input_2_right_cnn_r3_c2)

        # Feature extraction for layer 3
        input_1_right_cnn_r3_c3 = self.cnn_r3_c3(input_1_right_cnn_r3_c2)
        input_2_right_cnn_r4_c3 = self.cnn_r4_c3(input_2_right_cnn_r4_c2)

        # Concatenate
        concat_right_row_2 = torch.cat([input_2_right_cnn_r2_c1, input_1_right_cnn_r2_c2], dim=1)
        concat_right_row_3 = torch.cat([input_3_right_cnn_r3_c1, input_2_right_cnn_r3_c2, input_1_right_cnn_r3_c3], dim=1)
        concat_right_row_4 = torch.cat([input_4_right_cnn_r4_c1, input_3_right_cnn_r4_c2, input_2_right_cnn_r4_c3], dim=1)

        # Feature extraction right side output: 
        # * input_1_right_cnn_r1_c1
        # * concat_right_row_2
        # * concat_right_row_3
        # * concat_right_row_4
        
        # ------------- Warping features at each level
        # Calculate the flow for each level using the input of current level and the upsampled flow from the level + 1
        bfe_4_i1, bfe_4_i2, bfe_4_f_1_2, bfe_4_f_2_1 = self.flow_warp_r4(concat_left_row_4, concat_right_row_4, None, None)
        bfe_3_i1, bfe_3_i2, bfe_3_f_1_2, bfe_3_f_2_1 = self.flow_warp_r3(concat_left_row_3, concat_right_row_3, bfe_4_f_1_2, bfe_4_f_2_1)
        bfe_2_i1, bfe_2_i2, bfe_2_f_1_2, bfe_2_f_2_1 = self.flow_warp_r2(concat_left_row_2, concat_right_row_2, bfe_3_f_1_2, bfe_3_f_2_1)
        bfe_1_i1, bfe_1_i2, _, _ = self.flow_warp_r1(input_1_left_cnn_r1_c1, input_1_right_cnn_r1_c1, bfe_2_f_1_2, bfe_2_f_2_1)

        # Flow estimation output: 
        # * (bfe_1_i1, bfe_2_i1, bfe_3_i1, bfe_4_i1) 
        # * (bfe_1_i2, bfe_2_i2, bfe_3_i2, bfe_4_i2)
        
        # ------------- Warped features fusion   
        # Merge row 4
        add_row_4 = bfe_4_i1 + bfe_4_i2
        cnn_row_4 = self.cnn_r4(add_row_4)
        upsample_row_4 = self.up_r4(cnn_row_4)

        # Merge row 3
        add_row_3 = self.attention_r3(torch.cat([bfe_4_f_1_2, bfe_4_f_2_1], dim=1), bfe_3_i1 + bfe_3_i2) + upsample_row_4
        cnn_row_3 = self.cnn_r3(add_row_3)
        upsample_row_3 = self.up_r3(cnn_row_3)

        # Merge row 2
        add_row_2 = self.attention_r2(torch.cat([bfe_3_f_1_2, bfe_3_f_2_1], dim=1), bfe_2_i1 + bfe_2_i2) + upsample_row_3
        cnn_row_2 = self.cnn_r2(add_row_2)
        upsample_row_2 = self.up_r2(cnn_row_2)

        # Merge row 1
        add_row_1 = self.attention_r1(torch.cat([bfe_2_f_1_2, bfe_2_f_2_1], dim=1), bfe_1_i1 + bfe_1_i2) + upsample_row_2
        cnn_row_1 = self.cnn_r1(add_row_1)

        # Create the output layer
        fus_conv2d_outputs = self.act_out(self.cnn_out(cnn_row_1))
        
        # Feature fusion output: 
        # * fus_conv2d_outputs
        
        return fus_conv2d_outputs
