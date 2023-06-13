import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os

import torch.utils.data as data
import torch.optim as optim
import torchvision
import torch
import time
import pickle


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
        def __init__(self, resize=True):
            super(FBNetLoss.VGGPerceptualLoss, self).__init__()
            self.__name__ = "perceptual"
            blocks = []
            blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[:4].eval().to(device))
            blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[4:9].eval().to(device))
            blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[9:16].eval().to(device))
            blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[16:23].eval().to(device))
            for bl in blocks:
                for p in bl.parameters():
                    p.requires_grad = False
            self.blocks = torch.nn.ModuleList(blocks).to(device)
            self.transform = torch.nn.functional.interpolate
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
                    loss += torch.nn.functional.l1_loss(x, y)
                if i in style_layers:
                    act_x = x.reshape(x.shape[0], x.shape[1], -1)
                    act_y = y.reshape(y.shape[0], y.shape[1], -1)
                    gram_x = act_x @ act_x.permute(0, 2, 1)
                    gram_y = act_y @ act_y.permute(0, 2, 1)
                    loss += torch.nn.functional.l1_loss(gram_x, gram_y)
            return loss

    def __init__(self, pw=0.5, psnrw=1.0, msew=10.0, maew=5.0):
        super(FBNetLoss, self).__init__()
        self.pw = pw
        self.psnrw = psnrw
        self.msew = msew
        self.maew = maew
        self._perceptual_loss = FBNetLoss.VGGPerceptualLoss()
        self.__name__ = "loss"

    def forward(self, y_true, y_pred):
        perceptual_loss_ = self._perceptual_loss(y_true, y_pred)
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


def create_output_view(left, right, y, y_pred, figsize=(20, 4)):
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


def create_flow_view(left, right, y, y_pred, figsize=(20, 4)):
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


def create_attention_view(left, right, y, y_pred, figsize=(20, 4)):
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


def fit(model, train, valid, optimizer, loss, metrics, epochs, target_path, name, save_freq=500, log_freq=1, log_perf_freq=2500, mode="best"):  
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
            if step >= 10:  # todo: delete 
                break

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
                    filename = os.path.join(target_path, f'{name}_l={loss_avg}_e={epoch+1}_t={int(time.time())}.pth')
                    torch.save(model.state_dict(), filename)
                    
            # show the model performance
            if log_perf_freq is not None and step % log_perf_freq == 0 and step > 0:
                create_output_view(left, right, y, y_pred.detach())
                create_flow_view(model, left, right)
                create_attention_view(model, left, right)
                
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
            if step >= 10:  # todo: delete 
                break

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


def get_parser():
    parser = argparse.ArgumentParser(description='Training FBNet model')
    parser.add_argument("-ver", "--version", required=True, type=str, help="The version of the model")
    parser.add_argument("-n", "--name", required=False, type=str, default="fbnet", help="The name of the created model")
    parser.add_argument("-t", "--target", required=False, type=str, default="E:\\OneDrive - Akademia Górniczo-Hutnicza im. Stanisława Staszica w Krakowie\\Programming\\Labs\\Frame_booster\\models", help="The path where data is stored during the straining (such as history etc.)")
    parser.add_argument("-d", "--data", required=False, type=str, default="E:\\Data\\Video_Frame_Interpolation\\processed\\vimeo90k_pytorch", help="The source path of the dataset")
    parser.add_argument("-tr", "--train", required=False, type=str, default="train.txt", help="The name of file that contains training samples split")
    parser.add_argument("-v", "--valid", required=False, type=str, default="valid.txt", help="The source path of the validating dataset")
    parser.add_argument("-dev", "--device", required=False, type=str, default="gpu", help="The device used during the training (cpu or gpu)")
    parser.add_argument("-b", "--batch_size", required=False, type=int, default=2, help="The batch size")
    parser.add_argument("-e", "--epochs", required=False, type=int, default=10, help="The epochs count")
    parser.add_argument("-iw", "--width", required=False, type=int, default=256, help="The input image widht")
    parser.add_argument("-ih", "--height", required=False, type=int, default=144, help="The input image height")
    return parser.parse_args()


if __name__ == "__main__":
    # try:
    parser = get_parser()

    # verify arguments
    assert parser.version in ['v5', 'v6', 'v6_1'], f'Version {parser.version} is not currently implemented'
    assert parser.device in ['gpu', 'cpu'], "Device can only be set to gpu (cuda:0) or cpu"
    assert parser.name, "Name cannot be empty"
    assert parser.batch_size > 0, "Batch size cannot be negative"
    assert parser.epochs > 0, "Epochs cannot be negative"
    assert parser.width > 0, "Width cannot be negative"
    assert parser.height > 0, "Height cannot be negative"

    # select device
    if parser.device == 'gpu':
        if not torch.cuda.is_available():
            raise Exception("Cuda is not available")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # import proper model version
    if parser.version == "v5":
        import model_v5.modules as modules
    elif parser.version == "v6":
        import model_v6.modules as modules
    elif parser.version == "v6_1":
        import model_v6_1.modules as modules

    # check if dataset exists
    if not os.path.exists(parser.data):
        raise FileNotFoundError(f"Cannot find the following dir '{parser.data}'")
    if not os.path.exists(os.path.join(parser.data, 'data')):
        raise FileNotFoundError(f"Cannot find the following dir '{os.path.join(parser.data, 'data')}'")
    if not os.path.exists(os.path.join(parser.data, parser.train)):
        raise FileNotFoundError(f"Cannot find the following file '{os.path.join(parser.data, parser.train)}'")
    if not os.path.exists(os.path.join(parser.data, parser.valid)):
        raise FileNotFoundError(f"Cannot find the following file '{os.path.join(parser.data, parser.valid)}'")
    
    # check if destination dir exists
    if not os.path.exists(parser.target):
        raise FileNotFoundError(f"Cannot find the following dir '{parser.target}'")
    
    stamp = int(time.time())
    target_path = os.path.join(parser.target, f'model_{parser.version}', str(stamp))
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    # prepare the dataloader
    train_dataloader = data.DataLoader(
        dataset = ByteImageDataset(
            path = parser.data,
            subdir = 'data',
            split_filename = parser.train,
            shape = (parser.height, parser.width, 3)
        ),
        shuffle = True,
        batch_size = parser.batch_size,
        drop_last = True,
        prefetch_factor=10,
        num_workers=2
    )

    valid_dataloader = data.DataLoader(
        dataset = ByteImageDataset(
            path = parser.data,
            subdir = 'data',
            split_filename = parser.valid,
            shape = (parser.height, parser.width, 3)
        ),
        batch_size = parser.batch_size,
        drop_last = True,
        prefetch_factor=10,
        num_workers=2
    )

    # print the size of training and validating dataset
    print(f"Training {parser.name}_{parser.version}")
    print(f'Loaded the dataset from: "{parser.data}"')
    print(f'Data will be saved to: "{target_path}"')
    print(f'Training batches: {len(train_dataloader)}')
    print(f'Validating batches: {len(valid_dataloader)}')
    print(f'Batch size: {parser.batch_size}')
    print(f'Epochs: {parser.epochs}')
    print(f"Device: {device}")

    # create the model
    model = modules.FBNet(input_shape=(parser.batch_size, 3, parser.height, parser.width), device=device).to(device)
    optimizer = optim.NAdam(model.parameters(), lr=1e-4)
    loss = FBNetLoss()

    # train the model
    history = fit(
        model = model, 
        train = train_dataloader,
        valid = valid_dataloader,
        optimizer = optimizer, 
        loss = loss, 
        metrics = [loss.psnr],
        epochs = parser.epochs,
        target_path = target_path, 
        name = parser.name,
        save_freq = 1,
        log_freq = 1,
        log_perf_freq = 2500,
        mode = "best"
    )

    # save both the model and the history
    torch.save(model.state_dict(), os.path.join(target_path, f'{parser.name}_e={parser.epochs}_t={stamp}.pth'))
    with open(os.path.join(target_path, f'{parser.name}_history_e={parser.epochs}_t={stamp}.pickle'), 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # except Exception as e:
    #     print(e)
