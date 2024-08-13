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
    
def time_to_str(t, mode='min'):
  if mode == 'min':
    t = int(t) / 60
    hr = t // 60
    min = t % 60
    return '%2d hr %02d min' % (hr, min)
  elif mode == 'sec':
    t = int(t)
    min = t // 60
    sec = t % 60
    return '%2d min %02d sec' % (min, sec)
  else:
    raise NotImplementedError

