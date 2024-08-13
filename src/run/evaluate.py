from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.calibration import calibration_curve
import pandas as pd
import torch
import torch.nn as nn

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for features, labels in tqdm(iter(val_loader)):
            features = features.float().to(device)
            labels = labels.float().to(device)
            
            features, probs = model(features)
            
            loss = criterion(probs, labels)

            val_loss.append(loss.item())

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        _val_loss = np.mean(val_loss)

        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        
        # Calculate AUC score
        # auc_score = multiLabel_AUC(all_labels, all_probs)
    
    # return _val_loss, auc_score
    return _val_loss