from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
import wandb

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from config import CONFIG
from utils import *
from dataset import CustomDataset
from model import *

def train(model, optimizer, train_loader, val_loader, device, path):
    model.to(device)


    main_criterion = nn.BCELoss().to(device)
    if CONFIG.model == 'LCNN':
        if CONFIG.feat == 1:
            cent_criterion = CenterLoss(feat_dim=CONFIG.N_MFCC*2).to(device)
        if CONFIG.feat == 2:
            cent_criterion = CenterLoss(feat_dim=(CONFIG.n_mels//16*32)).to(device)
    if CONFIG.model == 'MLP':
            cent_criterion = CenterLoss(feat_dim=128).to(device)
    optimizer_centloss = torch.optim.SGD(cent_criterion.parameters(), lr=0.5)
    
    best_val_score = 0
    best_model = None
    
    for epoch in tqdm(range(1, CONFIG.N_EPOCHS+1), desc='Train Epoch'):
        model.train()
        train_loss = []
        center_loss = []
        xent_loss = []
        for features, labels in iter(train_loader):
            features = features.float().to(device)
            labels = labels.float().to(device)
            # print(features.shape)
            
            optimizer.zero_grad()
            
            features, output = model(features)
            # print(output.shape, labels.shape)
            main_loss = main_criterion(output, labels)
            cent_loss = cent_criterion(features, labels)

            cent_loss *= CONFIG.cent_loss_weight
            loss = main_loss + cent_loss
       
            optimizer_centloss.zero_grad()

            loss.backward()
            optimizer.step()
        
            for param in cent_criterion.parameters():
                param.grad.data *= (1. / CONFIG.cent_loss_weight)
            optimizer_centloss.step()
            
            train_loss.append(main_loss.item())
            xent_loss.append(loss.item())
            center_loss.append(cent_loss.item())
                    
        _val_loss, _val_score = validation(model, main_criterion, cent_criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        _center_loss = np.mean(center_loss)
        _xent_loss = np.mean(xent_loss)

        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Xent_loss : [{_xent_loss:.5f}] Center Loss : [{_center_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val AUC : [{_val_score:.5f}]')
        wandb.log({'Train_loss' : _train_loss, 'Xent_loss':_xent_loss, 'Center_loss':_center_loss, 'Val_loss' : _val_loss, 'Val_auc' : _val_score})
            
        if best_val_score < _val_score:
            best_val_score < _val_score
            best_model = model
            torch.save(model, f'{path}/ep_{epoch}_best.pt')
    
    return best_model

def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for features in tqdm(iter(test_loader)):
            features = features.float().to(device)
            
            _, probs = model(features)

            if probs.is_cuda:
                probs = probs.cpu().detach().numpy()
            else:
                probs = probs.detach().numpy()
            predictions += probs.tolist()
    return predictions

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    seed_everything(CONFIG.SEED) # Seed 고정
    date = get_date()   

    if CONFIG.train:
        wandb.init(
        # set the wandb project where this run will be logged
        project="antispoof",
        name = f'{date}_{CONFIG.user}',
        # track hyperparameters and run metadata
        config= {
        "model": CONFIG.model,
        "sr": CONFIG.SR,
        "feat": CONFIG.feat,
        "batch": CONFIG.BATCH_SIZE,
        "epoch": CONFIG.N_EPOCHS,
        "lr": CONFIG.LR,
        "seed":CONFIG.SEED,

            })
        
        
        df = pd.read_csv('data/train.csv')
        # df = df[:100]
        train_df, val_df, _, _ = train_test_split(df, df['label'], test_size=CONFIG.TEST_SIZE, random_state=CONFIG.SEED)

        if CONFIG.feat == 1:
            train_feat, train_labels = get_mfcc_feature(train_df, True)
            val_feat, val_labels = get_mfcc_feature(val_df, True)
            input_dim = CONFIG.N_MFCC
        if CONFIG.feat == 2:
            train_feat, train_labels = get_mstft_feature(train_df, True)
            val_feat, val_labels = get_mstft_feature(val_df, True)
            input_dim = CONFIG.n_mels
        

        train_dataset = CustomDataset(train_feat, train_labels)
        val_dataset = CustomDataset(val_feat, val_labels)

        train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG.BATCH_SIZE,
        shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG.BATCH_SIZE,
            shuffle=False
        )

        if CONFIG.model == 'MLP':
            model = MLP(input_dim, CONFIG.N_CLASSES)
        if CONFIG.model == 'LCNN':
            model = LCNN(input_dim, CONFIG.N_CLASSES)

        
        optimizer = torch.optim.Adam(params = model.parameters(), lr = CONFIG.LR)
        folder = f'ckpt/{date}'
        os.makedirs(folder, exist_ok=True)
        infer_model = train(model, optimizer, train_loader, val_loader, device, folder)
        print("MODEL READY!")

    if CONFIG.infer:
        if not infer_model:
            infer_model = torch.load(CONFIG.infer_model)
            print(f'{CONFIG.infer_model} successfully loaded!')

        test_df = pd.read_csv('data/test.csv')
        # test_df = test_df[:100]
        if CONFIG.feat == 1:
            test_feat = get_mfcc_feature(test_df, False)
        if CONFIG.feat == 2:
            test_feat = get_mstft_feature(test_df, False)
        test_dataset = CustomDataset(test_feat, None)
        print('dataset ready!')
        test_loader = DataLoader(
            test_dataset,
            batch_size=CONFIG.BATCH_SIZE,
            shuffle=False
        )

        preds = inference(infer_model, test_loader, device)

        submit = pd.read_csv('data/sample_submission.csv')
        submit.iloc[:, 1:] = preds
        print(submit.head())

        submit.to_csv(f'result/{date}.csv', index=False)


if __name__ == "__main__":
    main()

    