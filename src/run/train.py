import numpy as np
import wandb
import torch
import logging
from torch import nn
from tqdm import tqdm
import time
from timeit import default_timer as timer

import warnings
warnings.filterwarnings('ignore')

from src.run.evaluate import validation
from src.utils.loss import CenterLoss
from src.utils.utils import time_to_str
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()
def train(model, optimizer, train_loader, val_loader, device, args):

    logging.info('** Start Training ! **')
    logging.info(
        '--------|-- VALID --|-- TRAIN --|-- Current Best --|--------------|'
    )
    logging.info(
        ' epoch  |    loss   |    loss   |     val_loss     |    time      |'
    )
    logging.info(
        '------------------------------------------------------------------|'
    )

    start = timer()
    model.to(device)

    criterion = nn.BCELoss().to(device)
    if args.classifier == 'LCNN':
        if args.feature_extractor == 1:
            cent_criterion = CenterLoss(feat_dim=args.n_mfcc*2).to(device)
        if args.feature_extractor == 2:
            cent_criterion = CenterLoss(feat_dim=(args.n_mels//16*32)).to(device)
    if args.classifier == 'MLP':
            cent_criterion = CenterLoss(feat_dim=128).to(device)
    optimizer_centloss = torch.optim.SGD(cent_criterion.parameters(), lr=0.5)
    
    best_val_loss = 100
    best_model = None
    
    for epoch in tqdm(range(1, args.epochs+1), desc='Train Epoch'):
                    

        model.train()
        train_loss = []
        for features, labels in tqdm(iter(train_loader)):
            features = features.float().to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                output = model(features)
                loss = criterion(output, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss.append(loss.item())
                    
        # _val_loss, _auc, _brier, _ece, _val_score = validation(model, criterion, val_loader, device)
        # _train_loss = np.mean(train_loss)
        # print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val Combined : [{_val_score:.5f}]')
        # wandb.log({'Train_loss' : _train_loss, 'Val_loss' : _val_loss, 'Val_auc' : _auc, 'Val_brier' : _brier, 'Val_ece': _ece, 'Val_score': _val_score})
            
        # if best_val_score > _val_score:
        #     best_val_score = _val_score
        #     best_model = model

        _val_loss = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] ')
        wandb.log({'Train_loss' : _train_loss, 'Val_loss' : _val_loss, })
            
        if best_val_loss > _val_loss:
            best_val_loss = _val_loss
            best_model = model
            torch.save(model, f'{args.log_path}/ep_{epoch}_best.pt')

        logging.info(
                '  %4.1f |   %6.3f   |   %6.3f   |   %6.3f   | %s   '
                % (epoch, _val_loss, _train_loss, best_val_loss, time_to_str(timer() - start, 'min')))

        time.sleep(0.01)
    
    return best_model
