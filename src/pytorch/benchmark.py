import argparse
import os

import torch.utils.data as data
import torch
import time
import utils
import train


def get_parser():
    parser = argparse.ArgumentParser(description='Benchmark for FBNet models')
    parser.add_argument("-ver", "--version", action='append', required=True, type=str, help="The versions of benchmarked models")
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


def run(parser):
    # verify arguments
    for version in parser.version:
        assert version in ['v5', 'v6', 'v6_1', 'v6_2', 'v6_3', 'v6_4', 'v6_5', 'v6_6t', 'v6_6s', 'v6_7', 'v7', 'v7_1'], f'Version {version} is not currently implemented'
    assert parser.device in ['gpu', 'cpu'], "Device can only be set to gpu (cuda:0) or cpu"
    assert parser.name, "Name cannot be empty"
    assert parser.batch_size > 0, "Batch size cannot be negative"
    assert parser.epochs > 0, "Epochs cannot be negative"
    assert parser.width > 0, "Width cannot be negative"
    assert parser.height > 0, "Height cannot be negative"

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

    results = []
    for version in parser.version:
        # import proper model version
        if version == "v5":
            import model_v5.modules as modules
        elif version == "v6":
            import model_v6.modules as modules
        elif version == "v6_1":
            import model_v6_1.modules as modules
        elif version == "v6_2":
            import model_v6_2.modules as modules
        elif version == "v6_3":
            import model_v6_3.modules as modules
        elif version == "v6_4":
            import model_v6_4.modules as modules
        elif version == "v6_5":
            import model_v6_5.modules as modules
        elif version == "v6_6t":
            import model_v6_6t.modules as modules
        elif version == "v6_6s":
            import model_v6_6s.modules as modules
        elif version == "v6_7":
            import model_v6_7.modules as modules
        elif version == "v7":
            import model_v7.modules as modules
        elif version == "v7_1":
            import model_v7_1.modules as modules

        # create dir for the files
        target_path = os.path.join(parser.target, f'model_{version}')
        if not os.path.exists(target_path):
            os.mkdir(target_path)

        stamp = int(time.time())
        target_path = os.path.join(target_path, str(stamp))
        if not os.path.exists(target_path):
            os.mkdir(target_path)

        # launch training function
        score = train.train(
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            test_dataloader=test_dataloader,
            vis_batches=vis_batches,
            batch_size=parser.batch_size,
            epochs=parser.epochs,
            modules=modules,
            model_name=parser.name,
            model_version=version,
            data_path=parser.data,
            target_path=target_path,
            height=parser.height,
            width=parser.width,
            save_freq=parser.save_freq,
            log_freq=parser.log_freq,
            device=device
        )
        
        # save the result
        results.append((version, score))

        # print new line
        print()

    # sort the results by the score
    results.sort(key=lambda x: x[1])

    # print the results
    print("Results:")
    for result in results:
        print(f"{parser.name}_{result[0]}: test_loss={result[1]}")


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
    