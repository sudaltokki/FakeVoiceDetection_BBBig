from tqdm.auto import tqdm
import librosa
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.calibration import calibration_curve
import pandas as pd
import torch
import random
import os
import pickle
import torch.nn as nn
from datetime import datetime


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def multiLabel_AUC(y_true, y_scores):
    auc_scores = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        auc_scores.append(auc)
    mean_auc_score = np.mean(auc_scores)
    return mean_auc_score
    
# def validation(model, main_criterion, cent_criterion, val_loader, device):
#     model.eval()
#     val_loss, all_labels, all_probs = [], [], []
    
#     with torch.no_grad():
#         for features, labels in iter(val_loader):
#             features = features.float().to(device)
#             labels = labels.float().to(device)
            
#             features, probs = model(features)
            
#             cent_loss = cent_criterion(features, labels)
#             cent_loss *= args.cent_loss_weight
#             loss = main_criterion(probs, labels) + cent_loss

#             val_loss.append(loss.item())

#             all_labels.append(labels.cpu().numpy())
#             all_probs.append(probs.cpu().numpy())
        
#         _val_loss = np.mean(val_loss)

#         all_labels = np.concatenate(all_labels, axis=0)
#         all_probs = np.concatenate(all_probs, axis=0)

#         # Calculate AUC score
#         auc_score, brier_score, ece_score, combined_score = auc_brier_ece(all_labels, all_probs)

    
#     return _val_loss, auc_score, brier_score, ece_score, combined_score



def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    bin_totals = np.histogram(y_prob, bins=np.linspace(0, 1, n_bins + 1), density=False)[0]
    non_empty_bins = bin_totals > 0
    bin_weights = bin_totals / len(y_prob)
    bin_weights = bin_weights[non_empty_bins]
    prob_true = prob_true[:len(bin_weights)]
    prob_pred = prob_pred[:len(bin_weights)]
    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
    return ece
    
def auc_brier_ece(labels, probs):
    # Check for missing values in submission_df
    

    # Check if the number and names of columns are the same in both dataframes
    if len(labels) != len(probs):
        raise ValueError("The length of true labels and probs do not match.")

    
    # Calculate AUC for each class
    auc_scores = []
    for i in range(2):
        # print(labels[:, i], probs[:, i])
        auc = roc_auc_score(labels[:, i], probs[:, i])
        auc_scores.append(auc)


    # Calculate mean AUC
    mean_auc = np.mean(auc_scores)

    brier_scores = []
    ece_scores = []
    
    # Calculate Brier Score and ECE for each class
    for i in range(2):
        y_true, y_prob = labels[:, i], probs[:, i]
        # print(y_true, y_prob)
        # Brier Score
        brier = mean_squared_error(y_true, y_prob)
        brier_scores.append(brier)
        
        # ECE
        ece = expected_calibration_error(y_true, y_prob)
        ece_scores.append(ece)
    
    # Calculate mean Brier Score and mean ECE
    mean_brier = np.mean(brier_scores)
    mean_ece = np.mean(ece_scores)
    
    # Calculate combined score
    combined_score = 0.5 * (1 - mean_auc) + 0.25 * mean_brier + 0.25 * mean_ece
    
    return mean_auc, mean_brier, mean_ece, combined_score

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

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=2, feat_dim=128, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        # labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
    
def check_data(cfg):
    feat = {1: 'mfcc_feat', 2:'mstft_feat'}
    train_d = cfg['train_data']
    f = cfg['feat']
    sr = cfg['sr']
    g = list(cfg[feat[f]].values())[0]
    data_name = f'{train_d}_{f}_{g}_{sr}.pickle'
    if os.path.exists(data_name):
        with open(data_name, 'rb') as file:
            data = pickle.load(file)
            print(f"Data loaded from {data_name}")
            return data
    else:
        return False, data_name
