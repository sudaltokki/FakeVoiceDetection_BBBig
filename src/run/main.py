from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
import wandb
import sys
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from src.models.feature_extractor.mfcc_mstft import get_mfcc_feature, get_mstft_feature

from src.models.classifier.mlp import MLP
from src.models.classifier.lcnn import LCNN

from src.run.train import train
from src.run.test import inference

from src.utils.dataset import CustomDataset
from src.utils.params import parse_args
from src.utils.utils import seed_everything

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


def main(args):
    args = parse_args(args)

    infer_model = None
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    seed_everything(args.seed) # Seed 고정
    date = args.current_time 

    if args.train:

        if args.set_wandb:
            set_wandb(args)
        
        df = pd.read_csv(args.train_csv_path)
        df = df[:10]
        train_df, val_df, _, _ = train_test_split(df, df[['fake', 'real']], test_size=args.test_ratio, random_state=args.seed)

        if args.feature_extractor == 'mfcc':
            train_feat, train_labels = get_mfcc_feature(train_df, args, True)
            val_feat, val_labels = get_mfcc_feature(val_df, args, True)
            input_dim = args.n_mfcc
        elif args.feature_extractor == 'mstft':
            train_feat, train_labels = get_mstft_feature(train_df, args, True)
            val_feat, val_labels = get_mstft_feature(val_df, args, True)
            input_dim = args.n_mels
        
        # with open("train.pickle", "wb") as f:
        #     pickle.dump(train_feat, f)

        train_dataset = CustomDataset(train_feat, train_labels)
        val_dataset = CustomDataset(val_feat, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        if args.classifier == 'MLP':
            model = MLP(input_dim, args.n_classes)
        elif args.classifier == 'LCNN':
            model = LCNN(input_dim, args.n_classes)
        
        optimizer = torch.optim.Adam(params = model.parameters(), lr = args.lr)
        
        folder = f'ckpt/{date}'
        os.makedirs(folder, exist_ok=True)

        infer_model = train(model, optimizer, train_loader, val_loader, device, folder, args)
        print("MODEL READY!")

    if args.infer:
        if not infer_model:
            infer_model = torch.load(args.classifier_checkpoint)
            print(f'{args.classifier_checkpoint} successfully loaded!')

        test_df = pd.read_csv(args.test_csv_path)
        # test_df = test_df[:100]
        if args.feature_extractor == 'mfcc':
            test_feat = get_mfcc_feature(test_df, args, False)
        elif args.feature_extractor == 'mstft':
            test_feat = get_mstft_feature(test_df, args, False)
        test_dataset = CustomDataset(test_feat, None)
        print('dataset ready!')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        preds = inference(infer_model, test_loader, device)

        submit = pd.read_csv(args.submission_csv_path)
        submit.iloc[:, 1:] = preds
        print(submit.head())

        submit.to_csv(f'result/{date}.csv', index=False)


if __name__ == "__main__":
    main(sys.argv[1:])

    