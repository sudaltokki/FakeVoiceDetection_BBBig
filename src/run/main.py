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
from src.models.classifier.aasist import AASIST
from transformers import Wav2Vec2FeatureExtractor

from src.run.train import train
from src.run.inference import inference

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
        else:
            if args.feature_extractor == 'hubert':
                model_id = "facebook/hubert-base-ls960"
                print('feature_extractor is hubert base')
            elif args.feature_extractor == 'hubert-large':
                model_id = 'facebook/hubert-large-ls960-ft'
                print('feature_extractor is hubert large')
            elif args.feature_extractor == 'w2v2-xlsr-300m':
                model_id = 'facebook/wav2vec2-xls-r-300m'
                print('feature_extractor is w2v2')
            elif args.feature_extractor == 'wavlm-base-plus':
                model_id = 'microsoft/wavlm-base-plus'
                print('feature_extractor is wavlm')
            else: 
                print('choose right model')
                exit
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
       
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
        elif args.classifier == 'AASIST':
            aasist_args = {"first_conv": 128,
              "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
              "gat_dims": [64, 32],
              "pool_ratios": [0.5, 0.7, 0.5, 0.5],
              "temperatures": [2.0, 2.0, 100.0, 100.0]
              }
            
            model = AASIST(aasist_args)
            model.load_state_dict(
                    torch.load(args.aasist_ckpt, map_location=device))
            print("Model loaded : {}".format(args.aasist_ckpt))

        
        optimizer = torch.optim.Adam(params = model.parameters(), lr = args.lr, weight_decay=args.wd)

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
        print('test dataset ready!')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        preds = inference(infer_model, test_loader, device)

        submit = pd.read_csv(args.submission_csv_path)
        submit.iloc[:, 1:] = preds
        print(submit.head())

        submit.to_csv(f'result/{args.log_path}.csv', index=False)


if __name__ == "__main__":
    main(sys.argv[1:])

    