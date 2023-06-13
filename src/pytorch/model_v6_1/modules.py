import torch.nn.functional as F
import torch.nn as nn
import torch


class TReLU(nn.Module):
    def __init__(self, lower=0.0, upper=1.0, **kwargs):
        super(TReLU, self).__init__(**kwargs)
        self.lower = lower
        self.upper = upper

    def forward(self, x):
        return torch.clip(x, min=self.lower, max=self.upper)


class FlowFeatureWarp(nn.Module):
    def __init__(self, flow_input_chanels, use_norm=False, dropout=0.1, interpolation='bilinear',
                flow_info = {
                    "filter_counts": [32, 32, 48, 48, 64, 64, 80, 80, 48, 32],
                    "filter_sizes": [(9, 9), (9, 9), (7, 7), (7, 7), (5, 5), (5, 5), (3, 3), (3, 3), (1, 1), (1, 1)],
                    "filter_strides": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    "filter_paddings": [8, 4, 6, 3, 4, 2, 2, 1, 0, 0],
                    "filter_dilations": [2, 1, 2, 1, 2, 1, 2, 1, 1, 1]
                }, **kwargs):
        super(FlowFeatureWarp, self).__init__(**kwargs)
        
        fcounts = flow_info['filter_counts']
        fsizes = flow_info['filter_sizes']
        fstrides = flow_info['filter_strides']
        fpads = flow_info['filter_paddings']
        fdils = flow_info['filter_dilations']
        
        assert len(fcounts) == len(fsizes) == len(fstrides) == len(fpads) == len(fdils), "Filter options should have the same size"
        
        modules = []
        for fcount, fsize, fstride, fpad, fdil in zip(fcounts, fsizes, fstrides, fpads, fdils):
            modules.append(nn.Conv2d(flow_input_chanels, fcount, fsize, fstride, fpad, fdil))
            modules.append(nn.LeakyReLU(0.2))
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
            
            flow_input_chanels = fcount
        modules.append(nn.Conv2d(flow_input_chanels, 2, 3, 1, 1))
            
        self.flow = nn.Sequential(*modules)
        self.flow_coef = nn.Parameter(torch.ones(1))
        self.upsample = nn.Upsample(scale_factor=(2, 2), mode=interpolation)

    def forward(self, input_1, input_2, upsampled_flow):
        if torch.is_tensor(upsampled_flow):
            upsampled_flow = upsampled_flow * self.flow_coef
            input_2_warped_1 = FlowFeatureWarp.warp(input_2, upsampled_flow)
            flow_change = self.flow(torch.cat([input_1, input_2_warped_1], dim=1))
            flow_change = flow_change + upsampled_flow
        else:
            flow_change = self.flow(torch.cat([input_1, input_2], dim=1))
        
        input_2_warped_1 = FlowFeatureWarp.warp(input_2, flow_change * 0.5)
        flow_upsampled = self.upsample(flow_change)
        
        return input_2_warped_1, flow_upsampled
    
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
    def __init__(self, g_filters, x_filters, filters, interpolation='bilinear', **kwargs):
        super(AttentionGate, self).__init__(**kwargs)
        
        self.gcnn = nn.Conv2d(g_filters, filters, 3, 1, 1)
        self.xcnn = nn.Conv2d(x_filters, filters, 3, 2, 1)
        self.ocnn = nn.Conv2d(filters, 1, 1)
        self.upsample = nn.Upsample(scale_factor=(2, 2), mode=interpolation)
        self.act = nn.LeakyReLU(0.2)
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
    def __init__(self, input_shape, device, filters=[16, 32, 64, 64], dropout=0.1, interpolation="bilinear", 
                 flow_feature_warp = [
                     {
                        "filter_counts": [16, 24, 32, 40, 64, 64, 40, 40, 24, 24],
                        "filter_sizes": [(9, 9), (9, 9), (7, 7), (7, 7), (5, 5), (5, 5), (3, 3), (3, 3), (1, 1), (1, 1)],
                        "filter_strides": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        "filter_paddings": [8, 4, 6, 3, 4, 2, 2, 1, 0, 0],
                        "filter_dilations": [2, 1, 2, 1, 2, 1, 2, 1, 1, 1]
                     }, {
                        "filter_counts": [32, 40, 64, 72, 40, 40, 24, 24],
                        "filter_sizes": [(7, 7), (7, 7), (5, 5), (5, 5), (3, 3), (3, 3), (1, 1), (1, 1)],
                        "filter_strides": [1, 1, 1, 1, 1, 1, 1, 1],
                        "filter_paddings": [6, 3, 4, 2, 2, 1, 0, 0],
                        "filter_dilations": [2, 1, 2, 1, 2, 1, 1, 1]
                     }, {
                        "filter_counts": [32, 40, 40, 40, 24, 24],
                        "filter_sizes": [(5, 5), (5, 5), (3, 3), (3, 3), (1, 1), (1, 1)],
                        "filter_strides": [1, 1, 1, 1, 1, 1],
                        "filter_paddings": [4, 2, 2, 1, 0, 0],
                        "filter_dilations": [2, 1, 2, 1, 1, 1]
                     }, {
                        "filter_counts": [32, 40, 24, 24],
                        "filter_sizes": [(3, 3), (3, 3), (1, 1), (1, 1)],
                        "filter_strides": [1, 1, 1, 1],
                        "filter_paddings": [2, 1, 0, 0],
                        "filter_dilations": [2, 1, 1, 1]
                     }
                 ], **kwargs):
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
        
        self.act_r1_1 = nn.LeakyReLU(0.2)
        self.act_r1_2 = nn.LeakyReLU(0.2)
        self.act_r2_1 = nn.LeakyReLU(0.2)
        self.act_r2_2 = nn.LeakyReLU(0.2)
        self.act_r3_1 = nn.LeakyReLU(0.2)
        self.act_r3_2 = nn.LeakyReLU(0.2)
        self.act_r4_1 = nn.LeakyReLU(0.2)
        
        # ------------- Feature warping layers 
        self.flow_warp_r1 = FlowFeatureWarp(
            flow_input_chanels=2*filters[0], 
            dropout=dropout, 
            interpolation=interpolation,
            flow_info=flow_feature_warp[0]
        )
        self.flow_warp_r2 = FlowFeatureWarp(
            flow_input_chanels=2*filters[1], 
            dropout=dropout, 
            interpolation=interpolation,
            flow_info=flow_feature_warp[1]
        )
        self.flow_warp_r3 = FlowFeatureWarp(
            flow_input_chanels=2*filters[2], 
            dropout=dropout, 
            interpolation=interpolation,
            flow_info=flow_feature_warp[2]
        )
        self.flow_warp_r4 = FlowFeatureWarp(
            flow_input_chanels=2*filters[3], 
            dropout=dropout, 
            interpolation=interpolation,
            flow_info=flow_feature_warp[3]
        )
        
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
        
        self.act_r3_3 = nn.LeakyReLU(0.2)
        self.act_r2_3 = nn.LeakyReLU(0.2)
        self.act_r1_3 = nn.LeakyReLU(0.2)
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
