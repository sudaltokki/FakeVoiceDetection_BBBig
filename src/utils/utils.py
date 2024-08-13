from tqdm.auto import tqdm
import numpy as np
import torch
import wandb
import random
import sys
import os


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHseed'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def set_wandb(args):
    wandb_config= {
        "feature_extractor": args.feature_extractor,
        "classifier": args.classifier,
        "sr": args.sr,
        "feat": args.feature_extractor,
        "batch": args.batch_size,
        "epoch": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "train_data" : args.train_csv_path,
        "mfcc_feat" : {"n_mfcc":args.n_mfcc},
        "mstft_feat" : {"n_mels":args.n_mels, "n_fft":args.n_fft, "hop_len":args.hop_len, "win_len":args.win_len},
        }
    
    wandb.init(
            # set the wandb project where this run will be logged
            project="antispoof",
            name = f'{args.current_time}_{args.user}',
            # track hyperparameters and run metadata
            config= wandb_config
            )


    
def preprocess_spectrogram(spectrogram, max_length):
    if spectrogram.shape[-1] > max_length:
        n = spectrogram.shape[1]
        factor = n // max_length
        remainder = n % max_length

        downsampled_data = np.empty((spectrogram.shape[0], max_length))
        
        # 주된 부분에 대해 평균을 계산
        for i in range(max_length):
            start_index = i * factor
            end_index = start_index + factor
            downsampled_data[:, i] = spectrogram[:, start_index:end_index].mean(axis=1)
        
        # 남은 부분에 대해 평균을 계산
        if remainder != 0:
            downsampled_data[:, -1] = spectrogram[:, -remainder:].mean(axis=1)
        return downsampled_data
    else:
        pad_width = ((0, 0), (0, max_length - spectrogram.shape[1]))
        spectrogram = np.pad(spectrogram, pad_width, mode='constant')
    return spectrogram

