import torch.nn.functional as F
import torch.nn as nn
import torch


"""
This module contains fbnet v7
"""


class TReLU(nn.Module):
    def __init__(self, lower=0.0, upper=1.0, **kwargs):
        super(TReLU, self).__init__(**kwargs)
        self.lower = lower
        self.upper = upper

    def forward(self, x):
        return torch.clip(x, min=self.lower, max=self.upper)
    

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output=False, **kwargs):
        super(Conv2dBlock, self).__init__(**kwargs)
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.PReLU() if not output else TReLU()

    def forward(self, x):
        return self.act(self.cnn(x))
    

class FlowEstimator(nn.Module):
    def __init__(self, filters, fcount=32, fsize=5, **kwargs):
        super(FlowEstimator, self).__init__(**kwargs)
        
        # ------------- Flow estimation layers
        fpadding = fsize // 2
        self.cnn_1 = nn.Conv2d(2*filters, fcount, fsize, 1, fpadding)
        self.cnn_2 = nn.Conv2d(fcount, fcount, fsize, 1, fpadding)
        self.cnn_3 = nn.Conv2d(fcount, fcount, fsize, 1, fpadding)
        self.cnn_4 = nn.Conv2d(fcount, fcount, fsize, 1, fpadding)
        self.cnn_5 = nn.Conv2d(fcount, fcount, fsize, 1, fpadding)
        self.cnn_6 = nn.Conv2d(fcount, 2, fsize, 1, fpadding)

    def forward(self, source, target, flow):
        if torch.is_tensor(flow):
            x0 = FlowPyramid.warp(source, flow)
            x0 = torch.cat([x0, target], dim=1)
        else:
            x0 = torch.cat([source, target], dim=1)
        
        x1 = self.cnn_1(x0)
        x2 = self.cnn_2(x1) + x1
        x3 = self.cnn_3(x2) + x2
        x4 = self.cnn_4(x3) + x3
        x5 = self.cnn_5(x4) + x4
        x6 = self.cnn_6(x5)
        
        if torch.is_tensor(flow):
            return flow + x6
        else:
            return x6
        

class FlowPyramid(nn.Module):
    def __init__(self, in_channels, filters, fsizes, **kwargs):
        super(FlowPyramid, self).__init__(**kwargs)
        
        # ------------- Unet type encoding layers
        self.cnn_1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.cnn_2 = nn.Conv2d(in_channels, in_channels, 3, 2, 1)
        self.cnn_3 = nn.Conv2d(in_channels, in_channels + filters, 3, 1, 1)
        self.cnn_4 = nn.Conv2d(in_channels + filters, in_channels + filters, 3, 2, 1)
        self.cnn_5 = nn.Conv2d(in_channels + filters, in_channels + 2*filters, 3, 1, 1)
        self.cnn_6 = nn.Conv2d(in_channels + 2*filters, in_channels + 2*filters, 3, 2, 1)
        self.cnn_7 = nn.Conv2d(in_channels + 2*filters, in_channels + 3*filters, 3, 1, 1)
        
        # ------------- Flow estimation layers
        self.flow_1 = FlowEstimator(filters=in_channels, fcount=in_channels, fsize=fsizes[0])
        self.flow_2 = FlowEstimator(filters=in_channels + filters, fcount=in_channels + filters, fsize=fsizes[1])
        self.flow_3 = FlowEstimator(filters=in_channels + 2*filters, fcount=in_channels + 2*filters, fsize=fsizes[2])
        self.flow_4 = FlowEstimator(filters=in_channels + 3*filters, fcount=in_channels + 3*filters, fsize=fsizes[3])
        
        # ------------- Upsample layer
        self.upsample = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)
        
    def _encode(self, x):
        out1 = self.cnn_1(x)
        out2 = self.cnn_3(self.cnn_2(out1))
        out3 = self.cnn_5(self.cnn_4(out2))
        out4 = self.cnn_7(self.cnn_6(out3))
        return out1, out2, out3, out4
    
    def _process_flow(self, flow):
        return self.upsample(flow) * 2

    def forward(self, source, target):
        # make unet type encoded features
        s1, s2, s3, s4 = self._encode(source)
        t1, t2, t3, t4 = self._encode(target)
        
        # calculate flow
        f4 = self._process_flow(self.flow_4(s4, t4, None))
        f3 = self._process_flow(self.flow_3(s3, t3, f4))
        f2 = self._process_flow(self.flow_2(s2, t2, f3))
        f1 = self.flow_1(s1, t1, f2)

        return f1
    
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
    def __init__(self, input_shape, device, filters=16, fsizes=[3, 3, 3, 3], **kwargs):
        super(FBNet, self).__init__(**kwargs)
        
        self.b = input_shape[0]
        self.c = input_shape[1]
        self.h = input_shape[2]
        self.w = input_shape[3]
        
        # ------------- Feature encoding layers
        self.cnn_enblock_1 = Conv2dBlock(self.c, filters, 3, 1, 1)
        self.cnn_enblock_2 = Conv2dBlock(self.c + filters, filters, 3, 1, 1)
        self.cnn_enblock_3 = Conv2dBlock(self.c + 2*filters, filters, 3, 1, 1)
        self.cnn_enblock_4 = Conv2dBlock(self.c + 3*filters, filters, 3, 1, 1)
        
        # ------------- Feature decoding layers
        self.cnn_decblock_1 = Conv2dBlock(2*(self.c + 4*filters), 4*filters, 3, 1, 1)
        self.cnn_decblock_2 = Conv2dBlock(4*filters, 4*filters, 3, 1, 1)
        self.cnn_decblock_3 = Conv2dBlock(4*filters, 2*filters, 3, 1, 1)
        self.cnn_decblock_4 = Conv2dBlock(2*filters, 2*filters, 3, 1, 1)
        self.cnn_decblock_5 = Conv2dBlock(2*filters, 3, 3, 1, 1, output=True)
        
        # ------------- Flow pyramid layer
        self.flow_pyramid = FlowPyramid(in_channels=self.c + 4*filters, filters=filters, fsizes=fsizes)
        
    def _encode(self, x):
        x = torch.cat([self.cnn_enblock_1(x), x], dim=1)
        x = torch.cat([self.cnn_enblock_2(x), x], dim=1)
        x = torch.cat([self.cnn_enblock_3(x), x], dim=1)
        x = torch.cat([self.cnn_enblock_4(x), x], dim=1)
        return x
    
    def _flow(self, left, right):
        flow_left = self.flow_pyramid(left, right)
        flow_right = self.flow_pyramid(right, left)
        return flow_left, flow_right
    
    def _decode(self, left, right):
        x0 = torch.cat([left, right], dim=1)
        x1 = self.cnn_decblock_1(x0)
        x2 = self.cnn_decblock_2(x1) + x1
        x3 = self.cnn_decblock_3(x2)
        x4 = self.cnn_decblock_4(x3) + x3
        x5 = self.cnn_decblock_5(x4)
        return x5
    
    def forward(self, left, right):
        # encode features
        left = self._encode(left)
        right = self._encode(right)
        
        # calculate flow & warp features
        flow_left, flow_right = self._flow(left, right)
        left = FlowPyramid.warp(left, flow_left)
        right = FlowPyramid.warp(right, flow_right)
        
        # decode features
        result = self._decode(left, right)
        
#        # return the result
#         if self.training:
#             return result, flow_left
#         else:
#             return result
        # return the result
        return result
