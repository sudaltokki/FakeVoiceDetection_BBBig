from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
import wandb
import joblib

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from config import CONFIG, wandb_config
from utils import *
from dataset import CustomDataset
from models.model import *

def train(model, optimizer, train_loader, val_loader, device, path):
    model.to(device)
    main_criterion = nn.BCEWithLogitsLoss().to(device)
    if CONFIG.model == 'LCNN':
        if CONFIG.feat == 1:
            cent_criterion = CenterLoss(feat_dim=CONFIG.N_MFCC*2).to(device)
        if CONFIG.feat == 2:
            cent_criterion = CenterLoss(feat_dim=(CONFIG.n_mels//16*32)).to(device)
    if CONFIG.model == 'MLP':
            cent_criterion = CenterLoss(feat_dim=2).to(device)
    if CONFIG.model == 'RNET3':
            cent_criterion = CenterLoss(feat_dim=2).to(device)

    best_val_score = 1
    best_model = None
    scaler = GradScaler()
    
    for epoch in tqdm(range(1, CONFIG.N_EPOCHS+1), desc='Train Epoch'):
        model.train()
        train_loss = []
        center_loss = []
        xent_loss = []
        for features, labels in tqdm(iter(train_loader)):
            features = features.float().to(device)
            labels = labels.float().to(device)
            # print(features.shape)
            
            
            with autocast():
                features, output = model(features)
                # print(output.shape, labels.shape)
                # print(output)
                main_loss = main_criterion(output, labels)
                
            optimizer.zero_grad()
            
            cent_loss = cent_criterion(features, labels)
            cent_loss *= CONFIG.cent_loss_weight
            loss = main_loss + cent_loss

            scaler.scale(loss).backward()
        
            if CONFIG.cent_loss_weight != 0:
                for param in cent_criterion.parameters():
                    param.grad.data *= (0.5 / (CONFIG.cent_loss_weight * CONFIG.LR))
                # cent_scaler.scale(cent_loss).backward()
                # cent_scaler.step(optimizer_centloss)
                # cent_scaler.update()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss.append(main_loss.item())
            xent_loss.append(loss.item())
            center_loss.append(cent_loss.item())
                    
        _val_loss, _auc, _brier, _ece, _val_score = validation(model, main_criterion, cent_criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        _center_loss = np.mean(center_loss)
        _xent_loss = np.mean(xent_loss)

        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Xent_loss : [{_xent_loss:.5f}] Center Loss : [{_center_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val Combined : [{_val_score:.5f}]')
        wandb.log({'Train_loss' : _train_loss, 'Xent_loss':_xent_loss, 'Center_loss':_center_loss, 'Val_loss' : _val_loss, 'Val_auc' : _auc, 'Val_brier' : _brier, 'Val_ece': _ece, 'Val_score': _val_score})
            
        if best_val_score > _val_score:
            best_val_score = _val_score
            best_model = model
            torch.save(model.state_dict(), f'{path}/ep_{epoch}_best.pt')
    
    return best_model

def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for features in tqdm(iter(test_loader)):
            features = features.float().to(device)
            # print(features.shape)
            
            f, probs = model(features)
            # probs = torch.sigmoid(probs)

            if probs.is_cuda:
                probs = probs.cpu().detach().numpy()
            else:
                probs = probs.detach().numpy()
            predictions += probs.tolist()
    return predictions

def main():
    infer_model = None
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    seed_everything(CONFIG.SEED) # Seed 고정
    date = get_date() 

    if CONFIG.feat == 1:
        input_dim = CONFIG.N_MFCC
    if CONFIG.feat == 2:
        input_dim = CONFIG.n_mels

    if CONFIG.model == 'MLP':
        model = MLP(input_dim, CONFIG.N_CLASSES)
    if CONFIG.model == 'LCNN':
        model = LCNN(input_dim, CONFIG.N_CLASSES)
    # if CONFIG.model == 'RNET3':
    #     model = RawNet3(model_scale=8,
    #                     context=True,
    #                     summed=True,
    #                     encoder_type="ASP",
    #                     nOut=CONFIG.N_CLASSES,
    #                     out_bn=False,
    #                     sinc_stride=10,
    #                     log_sinc=True,
    #                     norm_sinc="mean",
    #                     grad_mult=1,
    #                 )  

    if CONFIG.train:
        wandb.init(
        # set the wandb project where this run will be logged
        project="antispoof",
        name = f'{date}_{CONFIG.user}',
        # track hyperparameters and run metadata
        config= wandb_config
        )
        
        df = pd.read_csv(CONFIG.TRAIN_PATH)
        df = df[:100]
        train_df, val_df, _, _ = train_test_split(df, df[['fake', 'real']], test_size=CONFIG.TEST_SIZE, random_state=CONFIG.SEED)

        data_exists = check_data(wandb_config)
        if data_exists[0] == False:
            train_feat, train_labels = get_feature(train_df, CONFIG.feat, CONFIG.mode, True)
            val_feat, val_labels = get_feature(val_df, CONFIG.feat, CONFIG.mode, True)

            feat_data = [train_feat, train_labels, val_feat, val_labels]
            joblib.dump(feat_data, data_exists[1])
            # with open(data_exists[1], "wb") as f:
            #     pickle.dump(feat_data, f)
            
        else:
            train_feat, train_labels, val_feat, val_labels = data_exists[0], data_exists[1], data_exists[2], data_exists[3]
            
        train_dataset = CustomDataset(train_feat, train_labels)
        val_dataset = CustomDataset(val_feat, val_labels)
        del data_exists

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
        
        optimizer = torch.optim.Adam(params = model.parameters(), lr = CONFIG.LR)
        folder = f'ckpt/{date}'
        os.makedirs(folder, exist_ok=True)
        infer_model = train(model, optimizer, train_loader, val_loader, device, folder)
        print("MODEL READY!")

    if CONFIG.infer:
        if not CONFIG.train:
            model.load_state_dict(torch.load(CONFIG.infer_model))
            infer_model = model
            # torch.save(infer_model.state_dict(), 'ep_4_best.pt')
            print(f'{CONFIG.infer_model} successfully loaded!')

        test_df = pd.read_csv('data/test.csv')
        # test_df = test_df[:100]
        test_feat = get_feature(test_df, CONFIG.feat, CONFIG.mode, False)
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

    