from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import logging
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
from src.utils.logger import setup_logging
from src.utils.params import parse_args
from src.utils.utils import seed_everything, set_wandb


def main(args):
    args = parse_args(args)

    infer_model = None
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    seed_everything(args.seed) # Seed 고정
    date = args.current_time 

    # set logger
    # log_base_path = os.path.join(
    #     args.log_path, 
    #     f'{args.current_time}_{args.feature_extractor}_{args.classifier}' + (f'_{args.extra_log}' if args.extra_log else '')
    # )
    # os.makedirs(log_base_path, exist_ok = True)
    # log_filename = 'out.log'
    # args.log_path = os.path.join(log_base_path, log_filename)
    # if os.path.exists(args.log_path):
    #     print("Error. Experiment already exists.")
    #     return -1
    
    # args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args)

    args_info = "\n".join([f"{arg}: {getattr(args, arg)}" for arg in vars(args)])
    logging.info(f"\n-----------------------------Arguments----------------------------|\n{args_info}\n")

    # training
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

        infer_model = train(model, optimizer, train_loader, val_loader, device, args)
        print("MODEL READY!")

    if args.infer:
        if not infer_model:
            infer_model = torch.load(args.classifier_checkpoint)
            print(f'{args.classifier_checkpoint} successfully loaded!')

        test_df = pd.read_csv(args.test_csv_path)
        test_df = test_df[:100]
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

    