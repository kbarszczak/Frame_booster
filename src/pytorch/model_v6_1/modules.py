# # Proof of concept notebook for the Frame Booster project
# - Author: Kamil Barszczak
# - Contact: kamilbarszczak62@gmail.com
# - Project: https://github.com/kbarszczak/Frame_booster

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import tqdm
import time
import cv2
import os

import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchsummary
import torchvision
import torch

base_path = 'E:/Data/Video_Frame_Interpolation/processed/vimeo90k_pytorch'
data_subdir = 'data'
train_ids = 'train.txt'
valid_ids = 'valid.txt'

width, height = 256, 144
epochs = 4
batch = 2

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

class ByteImageDataset(data.Dataset):
    def __init__(self, path, subdir, split_filename, shape):
        self.path = path
        self.subdir = subdir
        self.shape = shape
        self.ids = pd.read_csv(os.path.join(path, split_filename), names=["ids"])
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.subdir, str(self.ids.iloc[idx, 0]))
        
        imgs = [
            self._read_bytes_to_tensor(os.path.join(img_path, 'im1')),
            self._read_bytes_to_tensor(os.path.join(img_path, 'im3'))
        ]
        true = self._read_bytes_to_tensor(os.path.join(img_path, 'im2'))
        
        return imgs, true
    
    def _read_bytes_to_tensor(self, path):
        with open(path, 'rb') as bf:
            buf = bf.read()
            if len(buf) <= 0:
                print(F"Wrong buffer for {path}")
            return torch.permute(torch.reshape(torch.frombuffer(buf, dtype=torch.float), self.shape), (2, 0, 1))


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        self.__name__ = "perceptual"
        blocks = []
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[:4].eval().to(device))
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[4:9].eval().to(device))
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[9:16].eval().to(device))
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[16:23].eval().to(device))
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = nn.ModuleList(blocks).to(device)
        self.transform = F.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += F.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

    
def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))


def mse(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def psnr(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return 1 - psnr / 40.0


perceptual_loss = VGGPerceptualLoss()
    
    
def loss(y_true, y_pred):
    perceptual_loss_ = perceptual_loss(y_true, y_pred)
    psnr_ = psnr(y_true, y_pred)
    mse_ = mse(y_true, y_pred)
    mae_ = mae(y_true, y_pred)
    
    return 0.5*perceptual_loss_ + psnr_ + 5.0*mae_ + 10.0*mse_


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


class FBAttentionVNet(nn.Module):
    def __init__(self, input_shape, filters=[16, 32, 64, 64], dropout=0.1, interpolation="bilinear", 
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
        super(FBAttentionVNet, self).__init__(**kwargs)
        
        self.c = input_shape[0]
        self.h = input_shape[1]
        self.w = input_shape[2]
        
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


def plot_triplet(left, right, y, y_pred, figsize=(20, 4)):
    plt.figure(figsize=figsize)
    data = torch.cat([
        torchvision.transforms.functional.rotate(right, 90, expand=True),
        torchvision.transforms.functional.rotate(y_pred, 90, expand=True), 
        torchvision.transforms.functional.rotate(y, 90, expand=True), 
        torchvision.transforms.functional.rotate(left, 90, expand=True)
    ], dim=0)
    grid = torchvision.utils.make_grid(data, nrow=left.shape[0])
    grid = torchvision.transforms.functional.rotate(grid, 270, expand=True)
    plt.imshow(torch.permute(grid, (1, 2, 0)).cpu())
    plt.axis('off')
    plt.show()


def fit(model, train, valid, optimizer, loss, metrics, epochs, save_freq=500, log_freq=1, log_perf_freq=2500, mode="best"):  
    # create dict for a history
    history = {loss.__name__: []} | {metric.__name__: [] for metric in metrics} | {'val_' + loss.__name__: []} | {"val_" + metric.__name__: [] for metric in metrics}
    best_loss = None
    
    # loop over epochs
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")
        
        # create empty dict for loss and metrics
        loss_metrics = {loss.__name__: []} | {metric.__name__: [] for metric in metrics}

        # loop over training batches
        model.train(True)
        for step, record in enumerate(train):
            start = time.time()
            
            # extract the data
            left, right, y = record[0][0].to(device), record[0][1].to(device), record[1].to(device)

            # clear gradient
            model.zero_grad()
            
            # forward pass
            y_pred = model(left, right) 
            
            # calculate loss and apply the gradient
            loss_value = loss(y, y_pred)
            loss_value.backward()
            optimizer.step()
            
            # calculate metrics
            y_pred_detached = y_pred.detach()
            metrics_values = [metric(y, y_pred_detached) for metric in metrics]
            
            # save the loss and metrics
            loss_metrics[loss.__name__].append(loss_value.item())
            for metric, value in zip(metrics, metrics_values):
                loss_metrics[metric.__name__].append(value.item())
                
            end = time.time()
            
            # save the model
            if save_freq is not None and step % save_freq == 0 and step > 0:
                loss_avg = np.mean(loss_metrics[loss.__name__])
                if mode == "all" or (mode == "best" and (best_loss is None or best_loss > loss_avg)):
                    filename = f'../models/model_v6_1/fbnet_l={loss_avg}_e={epoch+1}_t={int(time.time())}.pth'
                    torch.save(model.state_dict(), filename)
                    
            # log the model performance
            if log_perf_freq is not None and step % log_perf_freq == 0 and step > 0:
                plot_triplet(left, right, y, y_pred.detach())
                
            # log the state
            if step % log_freq == 0:
                time_left = (end-start) * (len(train) - (step+1))
                print('\r[%5d/%5d] (eta: %s)' % ((step+1), len(train), time.strftime('%H:%M:%S', time.gmtime(time_left))), end='')
                for metric, values in loss_metrics.items():
                    print(f' {metric}=%.4f' % (np.mean(values)), end='')
            
        # save the training history
        for metric, values in loss_metrics.items():
            history[metric].extend(values)

        # setup dict for validation loss and metrics
        loss_metrics = {loss.__name__: []} | {metric.__name__: [] for metric in metrics}
        
        # process the full validating dataset
        model.train(False)
        for step, record in enumerate(valid):
            left, right, y = record[0][0].to(device), record[0][1].to(device), record[1].to(device)

            # forward pass
            y_pred = model(left, right).detach()
            
            # save the loss and metrics
            loss_metrics[loss.__name__].append(loss(y, y_pred).item())
            for metric, value in zip(metrics, [metric(y, y_pred) for metric in metrics]):
                loss_metrics[metric.__name__].append(value.item())
            
        # log the validation score & save the validation history
        for metric, values in loss_metrics.items():
            print(f' val_{metric}=%.4f' % (np.mean(values)), end='')
            history[f"val_{metric}"].extend(values)
            
        # restart state printer
        print()

    return history


if __name__ == "__main__":
    train_dataloader = data.DataLoader(
        dataset = ByteImageDataset(
            path = base_path,
            subdir = data_subdir,
            split_filename = train_ids,
            shape = (height, width, 3)
        ),
        shuffle = True,
        batch_size = batch,
        drop_last = True,
        prefetch_factor=20,
        num_workers=2
    )

    valid_dataloader = data.DataLoader(
        dataset = ByteImageDataset(
            path = base_path,
            subdir = data_subdir,
            split_filename = valid_ids,
            shape = (height, width, 3)
        ),
        batch_size = batch,
        drop_last = True,
        prefetch_factor=20,
        num_workers=2
    )

    print(f'Training batches: {len(train_dataloader)}')
    print(f'Validating batches: {len(valid_dataloader)}')

    fbnet = FBAttentionVNet(input_shape=(3, height, width)).to(device)

    history = fit(
        model = fbnet, 
        train = train_dataloader,
        valid = valid_dataloader,
        optimizer = optim.NAdam(fbnet.parameters(), lr=1e-4), 
        loss = loss, 
        metrics = [psnr],
        epochs = epochs, 
        save_freq = 500,
        log_freq = 1,
        log_perf_freq = None,
        mode = "best"
    )

    time_tmp = int(time.time())
    torch.save(fbnet.state_dict(), f'../models/model_v6_1/fbnet_e={epochs}_t={time_tmp}.pth')
    with open(f'../models/model_v6_1/history_t={time_tmp}.pickle', 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
