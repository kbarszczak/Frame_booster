import pandas as pd
import numpy as np
import argparse
import cv2
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
        def __init__(self):
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

    def __init__(self, pw=0.5, psnrw=1.0, msew=10.0, maew=5.0):
        super(FBNetLoss, self).__init__()
        self.pw = pw
        self.psnrw = psnrw
        self.msew = msew
        self.maew = maew
        self.perceptual_loss = FBNetLoss.VGGPerceptualLoss()
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


def create_output_view(model, batches, path):
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


def create_flow_view(model, batches, path):
    flows, names, hooks = {}, [], []
    
    # hook registration function
    def get_activation(name):
        def hook(model, input, output):
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


def create_attention_view(model, batches, path):
    attentions, names, hooks = {}, [], []
    
    # hook registration function
    def get_activation(name, act, upsample):
        def hook(model, input, output):
            attentions[name] = upsample(act(output.detach()))
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


def fit(model, train, valid, vis_batches, optimizer, loss, metrics, epochs, target_path, name, save_freq=500, log_freq=1, log_perf_freq=2500, mode="best"):  
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
                    filename = os.path.join(target_path, 'models', f'{name}_l={loss_avg}_e={epoch+1}_s={step+1}_t={int(time.time())}.pt')
                    torch.save(model.state_dict(), filename)
                    best_loss = loss_avg
                    
            # show the model performance
            if log_perf_freq is not None and step % log_perf_freq == 0 and step > 0:
                model.train(False)
                stamp = int(time.time())
                create_output_view(model, vis_batches, os.path.join(target_path, 'outputs', f'output_e={epoch+1}_s={step+1}_t={stamp}.png'))
                create_flow_view(model, vis_batches, os.path.join(target_path, 'flows', f'flow_e={epoch+1}_s={step+1}_t={stamp}.png'))
                create_attention_view(model, vis_batches, os.path.join(target_path, 'attentions', f'attention_e={epoch+1}_s={step+1}_t={stamp}.png'))
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


def get_parser():
    parser = argparse.ArgumentParser(description='Training FBNet model')
    parser.add_argument("-ver", "--version", required=True, type=str, help="The version of the model")
    parser.add_argument("-n", "--name", required=False, type=str, default="fbnet", help="The name of the created model")
    parser.add_argument("-t", "--target", required=False, type=str, default="..\\..\\tmp", help="The path where data is stored during the straining (such as history etc.)")
    parser.add_argument("-d", "--data", required=False, type=str, default="D:\\Data\\Video_Frame_Interpolation\\vimeo90k_pytorch", help="The source path of the dataset")
    parser.add_argument("-tr", "--train", required=False, type=str, default="train.txt", help="The name of file that contains training samples split")
    parser.add_argument("-v", "--valid", required=False, type=str, default="valid.txt", help="The name of file that contains validating samples split")
    parser.add_argument("-vis", "--visualization", required=False, type=str, default="vis.txt", help="The name of file that contains vaisualizing samples split")
    parser.add_argument("-dev", "--device", required=False, type=str, default="gpu", help="The device used during the training (cpu or gpu)")
    parser.add_argument("-b", "--batch_size", required=False, type=int, default=2, help="The batch size")
    parser.add_argument("-e", "--epochs", required=False, type=int, default=10, help="The epochs count")
    parser.add_argument("-iw", "--width", required=False, type=int, default=256, help="The input image widht")
    parser.add_argument("-ih", "--height", required=False, type=int, default=144, help="The input image height")
    return parser.parse_args()


def run(parser):
    # verify arguments
    assert parser.version in ['v5', 'v6', 'v6_1', 'v6_2', 'v6_3', 'v6_4', 'v6_5', 'v6_6'], f'Version {parser.version} is not currently implemented'
    assert parser.device in ['gpu', 'cpu'], "Device can only be set to gpu (cuda:0) or cpu"
    assert parser.name, "Name cannot be empty"
    assert parser.batch_size > 0, "Batch size cannot be negative"
    assert parser.epochs > 0, "Epochs cannot be negative"
    assert parser.width > 0, "Width cannot be negative"
    assert parser.height > 0, "Height cannot be negative"

    # import proper model version
    if parser.version == "v5":
        import model_v5.modules as modules
    elif parser.version == "v6":
        import model_v6.modules as modules
    elif parser.version == "v6_1":
        import model_v6_1.modules as modules
    elif parser.version == "v6_2":
        import model_v6_2.modules as modules
    elif parser.version == "v6_3":
        import model_v6_3.modules as modules
    elif parser.version == "v6_4":
        import model_v6_4.modules as modules
    elif parser.version == "v6_5":
        import model_v6_5.modules as modules
    elif parser.version == "v6_6":
        import model_v6_6.modules as modules

    # check if dataset exists
    if not os.path.exists(parser.data):
        raise FileNotFoundError(f"Cannot find the following dir '{parser.data}'")
    if not os.path.exists(os.path.join(parser.data, 'data')):
        raise FileNotFoundError(f"Cannot find the following dir '{os.path.join(parser.data, 'data')}'")
    if not os.path.exists(os.path.join(parser.data, 'vis')):
        raise FileNotFoundError(f"Cannot find the following dir '{os.path.join(parser.data, 'vis')}'")
    if not os.path.exists(os.path.join(parser.data, parser.train)):
        raise FileNotFoundError(f"Cannot find the following file '{os.path.join(parser.data, parser.train)}'")
    if not os.path.exists(os.path.join(parser.data, parser.valid)):
        raise FileNotFoundError(f"Cannot find the following file '{os.path.join(parser.data, parser.valid)}'")
    
    # check if destination dir exists
    if not os.path.exists(parser.target):
        raise FileNotFoundError(f"Cannot find the following dir '{parser.target}'")
    
    target_path = os.path.join(parser.target, f'model_{parser.version}')
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    stamp = int(time.time())
    target_path = os.path.join(target_path, str(stamp))
    if not os.path.exists(target_path):
        os.mkdir(target_path)
        os.mkdir(os.path.join(target_path, 'flows'))
        os.mkdir(os.path.join(target_path, 'models'))
        os.mkdir(os.path.join(target_path, 'attentions'))
        os.mkdir(os.path.join(target_path, 'outputs'))

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

    vis_dataloader = data.DataLoader(
        dataset = ByteImageDataset(
            path = parser.data,
            subdir = 'vis',
            split_filename = parser.visualization,
            shape = (parser.height, parser.width, 3)
        ),
        batch_size = parser.batch_size,
        drop_last = True
    )

    # prepare data for visualization
    vis_iterator = iter(vis_dataloader)
    vis_batches = [next(vis_iterator) for bi in range(len(vis_dataloader)) if bi in [0, 1, 2, 3, 4, 5]]

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
        vis_batches = vis_batches,
        optimizer = optimizer, 
        loss = loss.perceptual_loss, 
        metrics = [loss.psnr, loss.mse, loss.mae],
        epochs = parser.epochs,
        target_path = target_path, 
        name = parser.name,
        save_freq = 1000,
        log_freq = 1,
        log_perf_freq = 500,
        mode = "best"
    )

    # save both the model and the history
    torch.save(model.state_dict(), os.path.join(target_path, f'{parser.name}.pt'))
    with open(os.path.join(target_path, f'{parser.name}_history.pickle'), 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = get_parser()

    # select device
    if parser.device == 'gpu':
        if not torch.cuda.is_available():
            raise Exception("Cuda is not available")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    run(parser)
    