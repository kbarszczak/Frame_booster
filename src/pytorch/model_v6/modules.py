import torch.nn.functional as F
import torch.nn as nn
import torch


"""
This module contains fbnet v6 with the following features:
- Overall Attention U-Net with single feature warping
- Warp only one image
- Feedforward CNN flow prediciton architecture with learnable scale coef
- Flow concatenation not addition
- Attention gate usage
- Encoder with filters outputs on each level (without feature stacking)
- Decoder features concatenation
- Use of conv2d transpose to change size
"""


class FlowFeatureWarp(nn.Module):
    def __init__(self, flow_input_chanels,
                flow_info = {
                    "filter_counts": [32, 32, 64, 64, 96, 96, 64, 32],
                    "filter_sizes": [(7, 7), (7, 7), (5, 5), (5, 5), (3, 3), (3, 3), (1, 1), (1, 1)],
                    "filter_strides": [1, 1, 1, 1, 1, 1, 1, 1],
                    "filter_paddings": [6, 3, 4, 2, 2, 1, 0, 0],
                    "filter_dilations": [2, 1, 2, 1, 2, 1, 1, 1],
                    "activations": [nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU()],
                }, interpolation='bilinear', **kwargs):
        super(FlowFeatureWarp, self).__init__(**kwargs)
        
        fcounts = flow_info['filter_counts']
        fsizes = flow_info['filter_sizes']
        fstrides = flow_info['filter_strides']
        fpads = flow_info['filter_paddings']
        fdils = flow_info['filter_dilations']
        facts = flow_info['activations']
        
        assert len(fcounts) == len(fsizes) == len(fstrides) == len(fpads) == len(fdils) == len(facts), "Filter options should have the same size"
        
        modules = []
        for fcount, fsize, fstride, fpad, fdil, fact in zip(fcounts, fsizes, fstrides, fpads, fdils, facts):
            modules.append(nn.Conv2d(flow_input_chanels, fcount, fsize, fstride, fpad, fdil))
            modules.append(fact)
            flow_input_chanels = fcount
        modules.append(nn.Conv2d(flow_input_chanels, 2, 3, 1, 1))
            
        self.flow = nn.Sequential(*modules)
        self.flow_coef = nn.Parameter(torch.rand(1) + 0.5)
        self.upsample = nn.Upsample(scale_factor=(2, 2), mode=interpolation)

    def forward(self, input_1, input_2, upsampled_flow):
        upsampled_flow = upsampled_flow * self.flow_coef
        input_2_warped_1 = FlowFeatureWarp.warp(input_2, upsampled_flow)
        estimated_flow = self.flow(torch.cat([input_1, input_2_warped_1, upsampled_flow], dim=1))
        input_2_warped_1 = FlowFeatureWarp.warp(input_2, estimated_flow)
        
        return input_2_warped_1, self.upsample(estimated_flow)
    
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
    def __init__(self, g_filters, x_filters, filters, act=nn.PReLU(), out_act=nn.Sigmoid(), interpolation='bilinear', **kwargs):
        super(AttentionGate, self).__init__(**kwargs)
        
        self.gcnn = nn.Conv2d(g_filters, filters, 3, 1, 1)
        self.xcnn = nn.Conv2d(x_filters, filters, 3, 2, 1)
        self.ocnn = nn.Conv2d(filters, 1, 1)
        self.upsample = nn.Upsample(scale_factor=(2, 2), mode=interpolation)
        self.act = act
        self.out_act = out_act

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
    def __init__(self, 
                 input_shape, device,
                 filters = [32, 64, 80, 96], 
                 flow_feature_warp = [
                     {
                        "filter_counts": [32, 32, 64, 64, 96, 96, 64, 32],
                        "filter_sizes": [(7, 7), (7, 7), (5, 5), (5, 5), (3, 3), (3, 3), (1, 1), (1, 1)],
                        "filter_strides": [1, 1, 1, 1, 1, 1, 1, 1],
                        "filter_paddings": [6, 3, 4, 2, 2, 1, 0, 0],
                        "filter_dilations": [2, 1, 2, 1, 2, 1, 1, 1],
                        "activations": [nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU()],
                    }, {
                        "filter_counts": [64, 64, 96, 96, 64, 32],
                        "filter_sizes": [(5, 5), (5, 5), (3, 3), (3, 3), (1, 1), (1, 1)],
                        "filter_strides": [1, 1, 1, 1, 1, 1],
                        "filter_paddings": [4, 2, 2, 1, 0, 0],
                        "filter_dilations": [2, 1, 2, 1, 1, 1],
                        "activations": [nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU()],
                    }, {
                        "filter_counts": [80, 80, 128, 128, 80, 48],
                        "filter_sizes": [(5, 5), (5, 5), (3, 3), (3, 3), (1, 1), (1, 1)],
                        "filter_strides": [1, 1, 1, 1, 1, 1],
                        "filter_paddings": [4, 2, 2, 1, 0, 0],
                        "filter_dilations": [2, 1, 2, 1, 1, 1],
                        "activations": [nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU()],
                    }, {
                        "filter_counts": [96, 96, 80, 64],
                        "filter_sizes": [(3, 3), (3, 3), (1, 1), (1, 1)],
                        "filter_strides": [1, 1, 1, 1],
                        "filter_paddings": [2, 1, 0, 0],
                        "filter_dilations": [2, 1, 1, 1],
                        "activations": [nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU()],
                    }, 
                 ], interpolation="bilinear", **kwargs):
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
            flow_input_chanels=2*filters[0]+2, 
            flow_info=flow_feature_warp[0],
            interpolation=interpolation
        )
        self.flow_warp_r2 = FlowFeatureWarp(
            flow_input_chanels=2*filters[1]+2, 
            flow_info=flow_feature_warp[1],
            interpolation=interpolation
        )
        self.flow_warp_r3 = FlowFeatureWarp(
            flow_input_chanels=2*filters[2]+2, 
            flow_info=flow_feature_warp[2],
            interpolation=interpolation
        )
        self.flow_warp_r4 = FlowFeatureWarp(
            flow_input_chanels=2*filters[3]+2, 
            flow_info=flow_feature_warp[3],
            interpolation=interpolation
        )
        self.zero_flow = torch.zeros((self.b, 2, self.h//8, self.w//8), device=device)
        
        # ------------- Attention layers
        self.attention_r1 = AttentionGate(
            g_filters=filters[1], 
            x_filters=filters[0], 
            filters=filters[0], 
            interpolation=interpolation
        )
        self.attention_r2 = AttentionGate(
            g_filters=filters[2], 
            x_filters=filters[1], 
            filters=filters[1], 
            interpolation=interpolation
        )
        self.attention_r3 = AttentionGate(
            g_filters=filters[3], 
            x_filters=filters[2], 
            filters=filters[2], 
            interpolation=interpolation
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
        self.act_r1_4 = nn.Sigmoid()
        
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
        warp_r4, flow_r4 = self.flow_warp_r4(input_right_cnn_r4_1, input_left_cnn_r4_1, self.zero_flow)
        warp_r3, flow_r3 = self.flow_warp_r3(input_right_cnn_r3_1, input_left_cnn_r3_1, flow_r4)
        warp_r2, flow_r2 = self.flow_warp_r2(input_right_cnn_r2_1, input_left_cnn_r2_1, flow_r3)
        warp_r1, _ = self.flow_warp_r1(input_right_cnn_r1_1, input_left_cnn_r1_1, flow_r2)
        
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
        input_cnn_r1_4 = self.act_r1_4(self.cnn_r1_4(input_cnn_r1_3))
        
        return input_cnn_r1_4
