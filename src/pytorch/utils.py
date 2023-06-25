import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os

import torch.utils.data as data
import torchvision
import torch
import time


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
            return torch.from_numpy(np.transpose(np.reshape(np.frombuffer(bf.read(), dtype='float32'), self.shape), (2, 0, 1)).copy())
        

class FBNetLoss(torch.nn.Module):

    class VGGPerceptualLoss(torch.nn.Module):
        def __init__(self, device):
            super(FBNetLoss.VGGPerceptualLoss, self).__init__()
            self.__name__ = "perceptual"
            blocks = [
                torchvision.models.vgg16(weights='DEFAULT').features[:4].eval().to(device),
                torchvision.models.vgg16(weights='DEFAULT').features[4:9].eval().to(device),
                torchvision.models.vgg16(weights='DEFAULT').features[9:16].eval().to(device),
                torchvision.models.vgg16(weights='DEFAULT').features[16:23].eval().to(device)
            ]
            
            for block in blocks:
                for parameter in block.parameters():
                    parameter.requires_grad = False

            self.blocks = torch.nn.ModuleList(blocks).to(device)
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

        def forward(self, y_true, y_pred):
            y_true = (y_true-self.mean) / self.std
            y_pred = (y_pred-self.mean) / self.std
            loss = 0.0
            for block in self.blocks:
                y_true = block(y_true)
                y_pred = block(y_pred)
                loss += torch.nn.functional.l1_loss(y_true, y_pred)

            return loss

    def __init__(self, device, pw=0.5, psnrw=1.0, msew=10.0, maew=5.0):
        super(FBNetLoss, self).__init__()
        self.pw = pw
        self.psnrw = psnrw
        self.msew = msew
        self.maew = maew
        self.perceptual_loss = FBNetLoss.VGGPerceptualLoss(device)
        self.__name__ = "loss"

    def forward(self, y_true, y_pred):
        perceptual_loss_ = self.perceptual_loss(y_true, y_pred)
        psnr_ = self.psnr(y_true, y_pred)
        mse_ = self.mse(y_true, y_pred)
        mae_ = self.mae(y_true, y_pred)
        return self.pw*perceptual_loss_ + self.psnrw*psnr_ + self.maew*mae_ + self.msew*mse_
    
    def mae(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))

    def mse(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

    def psnr(self, y_true, y_pred):
        mse = torch.mean((y_true - y_pred) ** 2)
        psnr = 20 * torch.log10(1 / torch.sqrt(mse))
        return 1 - psnr / 40.0


def save_img(path, img):
    img = (img * 255).astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def create_output_view(model, batches, path, device):
    data, batch_size = None, batches[0][1].shape[0]
    for batch in batches:
        # forward pass (hooks register outputs)
        left, right, y = batch[0][0].to(device), batch[0][1].to(device), batch[1].to(device)
        y_pred = model(left, right).detach()

        # process each sample in the batch
        for index in range(batch_size):
            cat_list = [
                torch.unsqueeze(left[index, :, :, :], dim=0),
                torch.unsqueeze(y[index, :, :, :], dim=0),
                torch.unsqueeze(y_pred[index, :, :, :], dim=0),
                torch.unsqueeze(right[index, :, :, :], dim=0)
            ]
            if data is not None:
                cat_list = [data] + cat_list
            data = torch.cat(cat_list, dim=0)

    grid = torchvision.utils.make_grid(data, nrow=4)
    save_img(path, torch.permute(grid, (1, 2, 0)).cpu().numpy())


def create_flow_view(model, batches, path, device):
    flows, names, hooks = {}, [], []
    
    # hook registration function
    def get_activation(name):
        def hook(model, input, output):
            if torch.is_tensor(output):
                flows[name] = output.detach()
            else:
                flows[name] = output[-1].detach()
        return hook

    # register hooks
    for child in model.named_children():
        if "flow" in child[0]:
            hooks.append(child[1].register_forward_hook(get_activation(child[0])))
            names.append(child[0])

    # leave the function is there is not such layers
    if not hooks:
        return
    
    data, batch_size = None, batches[0][1].shape[0]
    resize = torchvision.transforms.Resize((batches[0][1].shape[2], batches[0][1].shape[3]), antialias=True)

    for batch in batches:
        # forward pass (hooks register outputs)
        left, right = batch[0][0].to(device), batch[0][1].to(device)
        _ = model(left, right).detach()

        # process each sample in the batch
        for index in range(batch_size):
            cat_list = [torch.unsqueeze(left[index, :, :, :], dim=0)]
            
            for name in names:
                flow = flows[name][index, :, :, :]
                flow = resize(torchvision.utils.flow_to_image(flow)) / 255.0
                cat_list.append(torch.unsqueeze(flow, dim=0))
                
            cat_list.append(torch.unsqueeze(right[index, :, :, :], dim=0))
            if data is not None:
                cat_list = [data] + cat_list
            data = torch.cat(cat_list, dim=0)

    # remove hooks
    for hook in hooks:
        hook.remove()

    # create the grid and save the data
    grid = torchvision.utils.make_grid(data, nrow=(2 + len(names)))
    save_img(path, torch.permute(grid, (1, 2, 0)).cpu().numpy())


def create_attention_view(model, batches, path, device):
    attentions, names, hooks = {}, [], []
    
    # hook registration function
    def get_activation(name, act, upsample):
        def hook(model, input, output):
            mask = act(output.detach())
            if upsample is not None:
                mask = upsample(mask)
            attentions[name] = mask
        return hook

    # register hooks
    for child in model.named_children():
        if "attention" in child[0]:
            hooks.append(child[1].ocnn.register_forward_hook(get_activation(child[0], child[1].out_act, child[1].upsample)))
            names.append(child[0])
            
    # leave the function is there is not such layers
    if not hooks:
        return
    
    data, batch_size = None, batches[0][1].shape[0]
    resize = torchvision.transforms.Resize((batches[0][1].shape[2], batches[0][1].shape[3]), antialias=True)
    gray2rgb = torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)

    for batch in batches:
        # forward pass (hooks register outputs)
        left, right = batch[0][0].to(device), batch[0][1].to(device)
        _ = model(left, right).detach()
        
        # process each sample in the batch
        for index in range(batch_size):
            cat_list = [torch.unsqueeze(left[index, :, :, :], dim=0)]
            
            for name in names:
                attention = attentions[name][index, :, :, :]
                attention = gray2rgb(resize(attention))
                cat_list.append(torch.unsqueeze(attention, dim=0))
                
            cat_list.append(torch.unsqueeze(right[index, :, :, :], dim=0))
            if data is not None:
                cat_list = [data] + cat_list
            data = torch.cat(cat_list, dim=0)

    # remove hooks
    for hook in hooks:
        hook.remove()

    # create the grid and save the data
    grid = torchvision.utils.make_grid(data, nrow=(2 + len(names)))
    save_img(path, torch.permute(grid, (1, 2, 0)).cpu().numpy())


def fit(model, train, valid, vis_batches, optimizer, loss, metrics, epochs, target_path, name, device, save_freq=500, log_freq=1, log_perf_freq=2500, mode="best"):  
    # create dict for a history
    history = {loss.__name__: []} | {metric.__name__: [] for metric in metrics} | {'val_' + loss.__name__: []} | {"val_" + metric.__name__: [] for metric in metrics}
    best_loss = None
    
    # loop over epochs
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")
        
        # create empty dict for loss and metrics
        loss_metrics = {loss.__name__: []} | {metric.__name__: [] for metric in metrics}

        # loop over batches
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
                    filename = os.path.join(target_path, f'{name}_l={loss_avg}_e={epoch+1}_s={step+1}.pt')
                    torch.save(model.state_dict(), filename)
                    best_loss = loss_avg
                    
            # show the model performance
            if log_perf_freq is not None and step % log_perf_freq == 0 and step > 0:
                model.train(False)
                create_output_view(model, vis_batches, os.path.join(target_path, f'output_e={epoch+1}_s={step+1}.png'), device)
                create_flow_view(model, vis_batches, os.path.join(target_path, f'flow_e={epoch+1}_s={step+1}.png'), device)
                create_attention_view(model, vis_batches, os.path.join(target_path, f'attention_e={epoch+1}_s={step+1}.png'), device)
                model.train(True)
                
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


def evaluate(model, test, loss, metrics, device):  
    # create empty dict for loss and metrics
    loss_metrics = {loss.__name__: 0} | {metric.__name__: 0 for metric in metrics}

    # loop over batches
    model.train(False)
    for step, record in enumerate(test):
        start = time.time()
        
        # extract the data
        left, right, y = record[0][0].to(device), record[0][1].to(device), record[1].to(device)
        
        # forward pass
        with torch.no_grad():
            y_pred = model(left, right)
        
        # calculate loss & metrics
        loss_metrics[loss.__name__] += loss(y, y_pred).item()
        for metric in metrics:
            loss_metrics[metric.__name__] += metric(y, y_pred).item()
            
        end = time.time()
            
        # log the state
        time_left = (end-start) * (len(test) - (step+1))
        print('\r[%5d/%5d] (eta: %s)' % ((step+1), len(test), time.strftime('%H:%M:%S', time.gmtime(time_left))), end='')
        for metric, value in loss_metrics.items():
            print(f' {metric}=%.4f' % (value/(step+1)), end='')
        
    # restart state printer
    print()

    return {metric: (value / len(test)) for metric, value in loss_metrics.items()}

def save_history_plot(path, history, figsize=(10,5), exclude=None, use_norm=True, steps=None):
    def compress_loss(loss, steps):
        return [np.average(loss[steps*i:steps*(i+1)]) for i in range(len(loss)//steps)]
    
    def norm(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    
    plt.clf()
    fig = plt.figure(figsize=figsize)
    
    history=history.copy()
    if exclude is not None:
        history = {metric: values for metric, values in history.items() if metric not in exclude}
        
    for metric, values in history.items():
        # skip val
        if "val" in metric:
            continue

        # make values equal size
        val_values = history[f'val_{metric}']
        if len(val_values) != len(values):
            gdc = gcd(len(val_values), len(values))
            values = compress_loss(values, len(values)//gdc)
            val_values = compress_loss(val_values, len(val_values)//gdc)

        if use_norm:
            pivot = len(values)
            data = norm(values + val_values)
            values = data[pivot:]
            val_values = data[:pivot]
            
        if steps is not None:
            values = compress_loss(values, steps)
            val_values = compress_loss(val_values, steps)
        
        history[metric] = values
        history[f'val_{metric}'] = val_values
        
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for index, (metric, values) in enumerate(history.items()):
        if "val" in metric:
            continue
        
        steps = range(1, len(values)+1)
        plt.plot(steps, values, colors[index], label=metric)
        plt.plot(steps, history[f'val_{metric}'], colors[index]+'--', label=f"val_{metric}")
    
    if not use_norm:
        plt.yticks(np.linspace(0, 5, 21))
        plt.ylim(0, 5)
    plt.title("Comparision of training and validating scores")
    plt.xlabel('Steps')
    plt.ylabel("Values" if not use_norm else "Values normalized")
    plt.legend(loc='upper left')
    plt.grid(True, axis='y')
    plt.savefig(path)
    plt.close(fig)