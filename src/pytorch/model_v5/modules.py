import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch


"""
This module contains fbnet v5 (implementation similiar to model FILM) with the following features:
- Overall U-Net with double feature warping
- Warp both images
- Feedforward CNN flow prediciton architecture
- Flow addition not concatenation
- Encoder with passing level information 2 layers below (see FILM implementation)
- Decoder features addition
"""


class FlowEstimation(nn.Module):
    def __init__(self, flow_input_chanels,
                flow_info = {
                    "filter_counts": [32, 64, 64, 16, 12, 12],
                    "filter_sizes": [(3, 3), (3, 3), (3, 3), (3, 3), (1, 1), (1, 1)],
                    "filter_strides": [1, 1, 1, 1, 1, 1],
                    "filter_padding": [1, 1, 1, 1, 0, 0],
                    "activations": [nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU()],
                }, **kwargs):
        super(FlowEstimation, self).__init__(**kwargs)

        modules = []
        last_output_size = flow_input_chanels
        for fcount, fsize, fstride, fpad, fact in zip(flow_info['filter_counts'], flow_info['filter_sizes'], flow_info['filter_strides'], flow_info['filter_padding'], flow_info['activations']):
            modules.append(nn.Conv2d(last_output_size, fcount, fsize, fstride, fpad))
            modules.append(fact)
            last_output_size = fcount

        modules.append(nn.Conv2d(last_output_size, 2, 1))
        self.flow = nn.Sequential(*modules)

    def forward(self, x):
        return self.flow(x)


class BidirectionalFeatureWarp(nn.Module):
    def __init__(self, flow_prediction, interpolation='bilinear', **kwargs):
        super(BidirectionalFeatureWarp, self).__init__(**kwargs)
        
        self.flow_prediction = flow_prediction
        self.flow_upsample_1_2 = nn.Upsample(scale_factor=(2, 2), mode=interpolation)
        self.flow_upsample_2_1 = nn.Upsample(scale_factor=(2, 2), mode=interpolation)

    def forward(self, input_1, input_2, flow_1_2, flow_2_1):
        if torch.is_tensor(flow_1_2) and torch.is_tensor(flow_2_1):
            input_1_warped_1 = BidirectionalFeatureWarp.warp(input_1, flow_1_2)
            input_2_warped_1 = BidirectionalFeatureWarp.warp(input_2, flow_2_1)
        else:
            input_1_warped_1 = input_1
            input_2_warped_1 = input_2
            
        flow_change_1_2_concat = torch.cat([input_2, input_1_warped_1], dim=1)
        flow_change_1_2 = self.flow_prediction(flow_change_1_2_concat)
        
        flow_change_2_1_concat = torch.cat([input_1, input_2_warped_1], dim=1)
        flow_change_2_1 = self.flow_prediction(flow_change_2_1_concat)
        
        if torch.is_tensor(flow_1_2) and torch.is_tensor(flow_2_1):
            flow_1_2_changed = flow_1_2 + flow_change_1_2
            flow_2_1_changed = flow_2_1 + flow_change_2_1
        else:
            flow_1_2_changed = flow_change_1_2
            flow_2_1_changed = flow_change_2_1
            
        input_1_warped_2 = BidirectionalFeatureWarp.warp(input_1, flow_1_2_changed)
        input_2_warped_2 = BidirectionalFeatureWarp.warp(input_2, flow_2_1_changed)
        flow_1_2_changed_upsampled = self.flow_upsample_1_2(flow_1_2_changed)
        flow_2_1_changed_upsampled = self.flow_upsample_2_1(flow_2_1_changed)
        
        return input_1_warped_2, input_2_warped_2, flow_1_2_changed_upsampled, flow_2_1_changed_upsampled
    
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
                    [96, 80, 64, 64],  # encoder_filters_col_1
                    [80, 64, 64],  # encoder_filters_col_2
                    [64, 64]  # encoder_filters_col_3
                ], 
                decoder_filters = [40, 40],  # decoder_filters
                flow_info = [
                    {  # flow_1
                        "filter_counts": [32, 48, 64, 80, 80, 48],
                        "filter_sizes": [7, 5, 5, 3, 1, 1],
                        "filter_strides": [1, 1, 1, 1, 1, 1],
                        "filter_padding": [3, 2, 2, 1, 0, 0],
                        "activations": [nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU()]
                    }, 
                    {  # flow_2
                        "filter_counts": [24, 32, 64, 64, 32],
                        "filter_sizes": [5, 3, 3, 1, 1],
                        "filter_strides": [1, 1, 1, 1, 1],
                        "filter_padding": [2, 1, 1, 0, 0],
                        "activations": [nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU()]
                    }, 
                    {  # flow_3
                        "filter_counts": [24, 48, 48, 16],
                        "filter_sizes": [3, 3, 1, 1],
                        "filter_strides": [1, 1, 1, 1],
                        "filter_padding": [1, 1, 0, 0],
                        "activations": [nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.PReLU()]
                    },
                ], interpolation="bilinear", **kwargs):
        super(FBNet, self).__init__(**kwargs)
        
        self.b = input_shape[0]
        self.c = input_shape[1]
        self.h = input_shape[2]
        self.w = input_shape[3]
        
        # ------------- Shared Conv2d, AvgPool2d & Resize encoding layers
        self.resize_1_2 = torchvision.transforms.Resize(size=(self.h//2, self.w//2), antialias=True)
        self.resize_1_4 = torchvision.transforms.Resize(size=(self.h//4, self.w//4), antialias=True)
        self.resize_1_8 = torchvision.transforms.Resize(size=(self.h//8, self.w//8), antialias=True)
        
        self.cnn_r1_c1 = nn.Conv2d(self.c, encoder_filters[0][0], 3, 1, 1)
        self.cnn_r2_c1 = nn.Conv2d(self.c, encoder_filters[0][1], 3, 1, 1)
        self.cnn_r3_c1 = nn.Conv2d(self.c, encoder_filters[0][2], 3, 1, 1)
        self.cnn_r4_c1 = nn.Conv2d(self.c, encoder_filters[0][3], 3, 1, 1)
        self.act_r1_c1 = nn.PReLU()
        self.act_r2_c1 = nn.PReLU()
        self.act_r3_c1 = nn.PReLU()
        self.act_r4_c1 = nn.PReLU()

        self.cnn_r2_c2 = nn.Conv2d(encoder_filters[0][0], encoder_filters[1][0], 3, 1, 1)
        self.cnn_r3_c2 = nn.Conv2d(encoder_filters[0][1], encoder_filters[1][1], 3, 1, 1)
        self.cnn_r4_c2 = nn.Conv2d(encoder_filters[0][2], encoder_filters[1][2], 3, 1, 1)
        self.act_r2_c2 = nn.PReLU()
        self.act_r3_c2 = nn.PReLU()
        self.act_r4_c2 = nn.PReLU()

        self.cnn_r3_c3 = nn.Conv2d(encoder_filters[1][0], encoder_filters[2][0], 3, 1, 1)
        self.cnn_r4_c3 = nn.Conv2d(encoder_filters[1][1], encoder_filters[2][1], 3, 1, 1)
        self.act_r3_c3 = nn.PReLU()
        self.act_r4_c3 = nn.PReLU()
        
        self.avg_r2_c1 = nn.AvgPool2d(2)
        self.avg_r3_c1 = nn.AvgPool2d(2)
        self.avg_r4_c1 = nn.AvgPool2d(2)
        
        self.avg_r3_c2 = nn.AvgPool2d(2)
        self.avg_r4_c2 = nn.AvgPool2d(2)
        
        # ------------- Feature warping layers 
        self.bidirectional_flow_warp_row_1 = BidirectionalFeatureWarp(
            flow_prediction = FlowEstimation(
                flow_input_chanels = 2*encoder_filters[0][0],
                flow_info = flow_info[0]
            ),
            interpolation = interpolation
        )
        self.bidirectional_flow_warp_row_2 = BidirectionalFeatureWarp(
            flow_prediction = FlowEstimation(
                flow_input_chanels = 2*(encoder_filters[0][1] + encoder_filters[1][0]),
                flow_info = flow_info[1]
            ),
            interpolation = interpolation
        )
        self.bidirectional_flow_warp_row_3 = BidirectionalFeatureWarp(
            flow_prediction = FlowEstimation(
                flow_input_chanels = 2*(encoder_filters[0][2] + encoder_filters[1][1] + encoder_filters[2][0]),
                flow_info = flow_info[2]
            ),
            interpolation = interpolation
        )
        
        # ------------- Decoding Conv2d layers
        self.cnn_r4_1 = nn.Conv2d(encoder_filters[0][3] + encoder_filters[1][2] + encoder_filters[2][1], encoder_filters[0][2] + encoder_filters[1][1] + encoder_filters[2][0], 3, 1, 1)
        self.act_r4_1 = nn.PReLU()
        self.up_r4 = nn.Upsample(scale_factor=(2, 2), mode=interpolation)
        
        self.cnn_r3_1 = nn.Conv2d(encoder_filters[0][2] + encoder_filters[1][1] + encoder_filters[2][0], encoder_filters[0][1] + encoder_filters[1][0], 3, 1, 1)
        self.act_r3_1 = nn.PReLU()
        self.up_r3 = nn.Upsample(scale_factor=(2, 2), mode=interpolation)
        
        self.cnn_r2_1 = nn.Conv2d(encoder_filters[0][1] + encoder_filters[1][0], encoder_filters[0][0], 3, 1, 1)
        self.cnn_r2_2 = nn.Conv2d(encoder_filters[0][0], encoder_filters[0][0], 3, 1, 1)
        self.act_r2_1 = nn.PReLU()
        self.act_r2_2 = nn.PReLU()
        self.up_r2 = nn.Upsample(scale_factor=(2, 2), mode=interpolation)
        
        self.cnn_r1_1 = nn.Conv2d(encoder_filters[0][0], decoder_filters[0], 3, 1, 1)
        self.cnn_r1_2 = nn.Conv2d(decoder_filters[0], decoder_filters[1], 3, 1, 1)
        self.act_r1_1 = nn.PReLU()
        self.act_r1_2 = nn.PReLU()
        
        self.cnn_out = nn.Conv2d(decoder_filters[1], 3, 1, 1, 0)
        self.act_out = nn.Sigmoid()

    def forward(self, input_1_left, input_1_right):
        # ------------- Process left input
        input_2_left = self.resize_1_2(input_1_left)
        input_3_left = self.resize_1_4(input_2_left)
        input_4_left = self.resize_1_8(input_3_left)
        
        # Feature extraction for layer 1
        input_1_left_cnn_r1_c1 = self.act_r1_c1(self.cnn_r1_c1(input_1_left))
        input_2_left_cnn_r2_c1 = self.act_r2_c1(self.cnn_r2_c1(input_2_left))
        input_3_left_cnn_r3_c1 = self.act_r3_c1(self.cnn_r3_c1(input_3_left))
        input_4_left_cnn_r4_c1 = self.act_r4_c1(self.cnn_r4_c1(input_4_left))

        # Downsample layer 1
        input_1_left_cnn_r2_c1 = self.avg_r2_c1(input_1_left_cnn_r1_c1)
        input_2_left_cnn_r3_c1 = self.avg_r3_c1(input_2_left_cnn_r2_c1)
        input_3_left_cnn_r4_c1 = self.avg_r4_c1(input_3_left_cnn_r3_c1)

        # Feature extraction for layer 2
        input_1_left_cnn_r2_c2 = self.act_r2_c2(self.cnn_r2_c2(input_1_left_cnn_r2_c1))
        input_2_left_cnn_r3_c2 = self.act_r3_c2(self.cnn_r3_c2(input_2_left_cnn_r3_c1))
        input_3_left_cnn_r4_c2 = self.act_r4_c2(self.cnn_r4_c2(input_3_left_cnn_r4_c1))

        # Downsample layer 2
        input_1_left_cnn_r3_c2 = self.avg_r3_c2(input_1_left_cnn_r2_c2)
        input_2_left_cnn_r4_c2 = self.avg_r4_c2(input_2_left_cnn_r3_c2)

        # Feature extraction for layer 3
        input_1_left_cnn_r3_c3 = self.act_r3_c3(self.cnn_r3_c3(input_1_left_cnn_r3_c2))
        input_2_left_cnn_r4_c3 = self.act_r4_c3(self.cnn_r4_c3(input_2_left_cnn_r4_c2))

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
        input_1_right_cnn_r1_c1 = self.act_r1_c1(self.cnn_r1_c1(input_1_right))
        input_2_right_cnn_r2_c1 = self.act_r2_c1(self.cnn_r2_c1(input_2_right))
        input_3_right_cnn_r3_c1 = self.act_r3_c1(self.cnn_r3_c1(input_3_right))
        input_4_right_cnn_r4_c1 = self.act_r4_c1(self.cnn_r4_c1(input_4_right))

        # Downsample layer 1
        input_1_right_cnn_r2_c1 = self.avg_r2_c1(input_1_right_cnn_r1_c1)
        input_2_right_cnn_r3_c1 = self.avg_r3_c1(input_2_right_cnn_r2_c1)
        input_3_right_cnn_r4_c1 = self.avg_r4_c1(input_3_right_cnn_r3_c1)

        # Feature extraction for layer 2
        input_1_right_cnn_r2_c2 = self.act_r2_c2(self.cnn_r2_c2(input_1_right_cnn_r2_c1))
        input_2_right_cnn_r3_c2 = self.act_r3_c2(self.cnn_r3_c2(input_2_right_cnn_r3_c1))
        input_3_right_cnn_r4_c2 = self.act_r4_c2(self.cnn_r4_c2(input_3_right_cnn_r4_c1))

        # Downsample layer 2
        input_1_right_cnn_r3_c2 = self.avg_r3_c2(input_1_right_cnn_r2_c2)
        input_2_right_cnn_r4_c2 = self.avg_r4_c2(input_2_right_cnn_r3_c2)

        # Feature extraction for layer 3
        input_1_right_cnn_r3_c3 = self.act_r3_c3(self.cnn_r3_c3(input_1_right_cnn_r3_c2))
        input_2_right_cnn_r4_c3 = self.act_r4_c3(self.cnn_r4_c3(input_2_right_cnn_r4_c2))

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
        bfe_4_i1, bfe_4_i2, bfe_4_f_1_2, bfe_4_f_2_1 = self.bidirectional_flow_warp_row_3(concat_left_row_4, concat_right_row_4, None, None)
        bfe_3_i1, bfe_3_i2, bfe_3_f_1_2, bfe_3_f_2_1 = self.bidirectional_flow_warp_row_3(concat_left_row_3, concat_right_row_3, bfe_4_f_1_2, bfe_4_f_2_1)
        bfe_2_i1, bfe_2_i2, bfe_2_f_1_2, bfe_2_f_2_1 = self.bidirectional_flow_warp_row_2(concat_left_row_2, concat_right_row_2, bfe_3_f_1_2, bfe_3_f_2_1)
        bfe_1_i1, bfe_1_i2, _, _ = self.bidirectional_flow_warp_row_1(input_1_left_cnn_r1_c1, input_1_right_cnn_r1_c1, bfe_2_f_1_2, bfe_2_f_2_1)

        # Flow estimation output: 
        # * (bfe_1_i1, bfe_2_i1, bfe_3_i1, bfe_4_i1) 
        # * (bfe_1_i2, bfe_2_i2, bfe_3_i2, bfe_4_i2)
        
        # ------------- Warped features fusion   
        # Merge row 4
        add_row_4 = bfe_4_i1 + bfe_4_i2
        cnn_row_4_1 = self.act_r4_1(self.cnn_r4_1(add_row_4))
        upsample_row_4 = self.up_r4(cnn_row_4_1)

        # Merge row 3
        add_row_3 = bfe_3_i1 + bfe_3_i2 + upsample_row_4
        cnn_row_3_1 = self.act_r3_1(self.cnn_r3_1(add_row_3))
        upsample_row_3 = self.up_r3(cnn_row_3_1)

        # Merge row 2
        add_row_2 = bfe_2_i1 + bfe_2_i2 + upsample_row_3
        cnn_row_2_1 = self.act_r2_1(self.cnn_r2_1(add_row_2))
        cnn_row_2_2 = self.act_r2_2(self.cnn_r2_2(cnn_row_2_1))
        upsample_row_2 = self.up_r2(cnn_row_2_2)

        # Merge row 1
        add_row_1 = bfe_1_i1 + bfe_1_i2 + upsample_row_2
        cnn_row_1_1 = self.act_r1_1(self.cnn_r1_1(add_row_1))
        cnn_row_1_2 = self.act_r1_2(self.cnn_r1_2(cnn_row_1_1))

        # Create the output layer
        fus_conv2d_outputs = self.act_out(self.cnn_out(cnn_row_1_2))
        
        # Feature fusion output: 
        # * fus_conv2d_outputs
        
        return fus_conv2d_outputs
