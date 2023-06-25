import argparse
import os

import torch.utils.data as data
import torch.optim as optim
import torch
import time
import pickle
import utils


def get_parser():
    parser = argparse.ArgumentParser(description='Training FBNet model')
    parser.add_argument("-ver", "--version", required=True, type=str, help="The version of the model")
    parser.add_argument("-n", "--name", required=False, type=str, default="fbnet", help="The name of the created model")
    parser.add_argument("-t", "--target", required=False, type=str, default="..\\..\\tmp", help="The path where data is stored during the straining (such as history etc.)")
    parser.add_argument("-d", "--data", required=False, type=str, default="D:\\Data\\Video_Frame_Interpolation\\vimeo90k_pytorch", help="The source path of the dataset")
    parser.add_argument("-tr", "--train", required=False, type=str, default="train.txt", help="The name of file that contains training samples split")
    parser.add_argument("-ts", "--test", required=False, type=str, default="test.txt", help="The name of file that contains testing samples split")
    parser.add_argument("-v", "--valid", required=False, type=str, default="valid.txt", help="The name of file that contains validating samples split")
    parser.add_argument("-vis", "--visualization", required=False, type=str, default="vis.txt", help="The name of file that contains vaisualizing samples split")
    parser.add_argument("-dev", "--device", required=False, type=str, default="gpu", help="The device used during the training (cpu or gpu)")
    parser.add_argument("-b", "--batch_size", required=False, type=int, default=2, help="The batch size")
    parser.add_argument("-e", "--epochs", required=False, type=int, default=10, help="The epochs count")
    parser.add_argument("-iw", "--width", required=False, type=int, default=256, help="The input image widht")
    parser.add_argument("-ih", "--height", required=False, type=int, default=144, help="The input image height")
    parser.add_argument("-lf", "--log_freq", required=False, type=int, default=500, help="The steps to log model performance")
    parser.add_argument("-sf", "--save_freq", required=False, type=int, default=1000, help="The steps to save the model state")
    return parser.parse_args()


def train(train_dataloader, valid_dataloader, test_dataloader, vis_batches, batch_size, epochs, modules, model_name, model_version, data_path, target_path, height, width, save_freq, log_freq, device):
    # print the size of training and validating dataset
    print(f"Training {model_name}_{model_version}")
    print(f'Loaded the dataset from: "{data_path}"')
    print(f'Data will be saved to: "{target_path}"')
    print(f'Training batches: {len(train_dataloader)}')
    print(f'Validating batches: {len(valid_dataloader)}')
    print(f'Batch size: {batch_size}')
    print(f'Epochs: {epochs}')
    print(f"Device: {device}")

    # create the model
    model = modules.FBNet(input_shape=(batch_size, 3, height, width), device=device).to(device)
    optimizer = optim.NAdam(model.parameters(), lr=1e-4)
    loss = utils.FBNetLoss(device=device)

    # train the model
    history = utils.fit(
        model = model, 
        train = train_dataloader,
        valid = valid_dataloader,
        vis_batches = vis_batches,
        optimizer = optimizer, 
        loss = loss, 
        metrics = [loss.psnr, loss.mse, loss.mae],
        epochs = epochs,
        target_path = target_path, 
        name = model_name,
        device = device,
        save_freq = save_freq,
        log_freq = 1,
        log_perf_freq = log_freq,
        mode = "best"
    )

    # evaluate the training
    score = utils.evaluate(
        model=model,
        test=test_dataloader,
        loss=loss,
        metrics=[loss.psnr, loss.mse, loss.mae],
        device=device
    )

    # save the model, the history, and the testing score
    history_path = os.path.join(target_path, f'{model_name}_history.pickle')
    model_path = os.path.join(target_path, f'{model_name}.pt')
    test_path = os.path.join(target_path, f'{model_name}_score.txt')
    torch.save(model.state_dict(), model_path)
    with open(history_path, 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(test_path, 'w') as f:
        print(score, file=f)

    # save the plots of history
    plot_5_path = os.path.join(target_path, f'plot_5.png')
    plot_5_loss_path = os.path.join(target_path, f'plot_5_loss.png')
    plot_5_raw_loss_path = os.path.join(target_path, f'plot_5_raw_loss.png')
    plot_50_path = os.path.join(target_path, f'plot_50.png')
    plot_50_loss_path = os.path.join(target_path, f'plot_50_loss.png')
    plot_50_raw_loss_path = os.path.join(target_path, f'plot_50_raw_loss.png')
    utils.save_history_plot(plot_5_path, history, exclude=None, use_norm=True, steps=5)
    utils.save_history_plot(plot_5_loss_path, history, exclude=['psnr', 'mse', 'mae', 'ssim'], use_norm=True, steps=5)
    utils.save_history_plot(plot_5_raw_loss_path, history, exclude=['psnr', 'mse', 'mae', 'ssim'], use_norm=False, steps=5)
    utils.save_history_plot(plot_50_path, history, exclude=None, use_norm=True, steps=50)
    utils.save_history_plot(plot_50_loss_path, history, exclude=['psnr', 'mse', 'mae', 'ssim'], use_norm=True, steps=50)
    utils.save_history_plot(plot_50_raw_loss_path, history, exclude=['psnr', 'mse', 'mae', 'ssim'], use_norm=False, steps=50)

    # Log training results
    print(f"Training of {model_name}_{model_version} finished")
    print(f"Testing score:")
    for m, v in score.items():
        print(f'{m}: {v}')
    print(f"Trained model saved to {model_path}")
    print(f"Training history saved to {history_path}")
    print(f"Testing score saved to {test_path}")
    print('Plots saved to the following files:')
    print(plot_5_path)
    print(plot_5_loss_path)
    print(plot_5_raw_loss_path)
    print(plot_50_path)
    print(plot_50_loss_path)
    print(plot_50_raw_loss_path)

    return score[loss.__name__]


def run(parser):
    # verify arguments
    assert parser.version in ['v5', 'v6', 'v6_1', 'v6_2', 'v6_3', 'v6_4', 'v6_5', 'v6_6t', 'v6_6s', 'v6_7', 'v7'], f'Version {parser.version} is not currently implemented'
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
    elif parser.version == "v6_6t":
        import model_v6_6t.modules as modules
    elif parser.version == "v6_6s":
        import model_v6_6s.modules as modules
    elif parser.version == "v6_7":
        import model_v6_7.modules as modules
    elif parser.version == "v7":
        import model_v7.modules as modules

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

    # prepare the dataloaders
    train_dataloader = data.DataLoader(
        dataset = utils.ByteImageDataset(
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

    test_dataloader = data.DataLoader(
        dataset = utils.ByteImageDataset(
            path = parser.data,
            subdir = 'data',
            split_filename = parser.test,
            shape = (parser.height, parser.width, 3)
        ),
        shuffle = True,
        batch_size = parser.batch_size,
        drop_last = True,
        prefetch_factor=10,
        num_workers=2
    )

    valid_dataloader = data.DataLoader(
        dataset = utils.ByteImageDataset(
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
        dataset = utils.ByteImageDataset(
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
    vis_batches = [next(vis_iterator) for bi in range(len(vis_dataloader)) if bi in [0, 1, 2]]

    # launch training function
    _ = train(
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        test_dataloader=test_dataloader,
        vis_batches=vis_batches,
        batch_size=parser.batch_size,
        epochs=parser.epochs,
        modules=modules,
        model_name=parser.name,
        model_version=parser.version,
        data_path=parser.data,
        target_path=target_path,
        height=parser.height,
        width=parser.width,
        save_freq=parser.save_freq,
        log_freq=parser.log_freq,
        device=device
    )


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
    