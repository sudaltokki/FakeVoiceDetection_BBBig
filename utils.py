from tqdm.auto import tqdm
import librosa
from config import CONFIG
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
import joblib

def get_date():
    # 현재 날짜와 시간을 가져옴
    now = datetime.now()
    # 원하는 형식으로 변환
    current_datetime_str = now.strftime("%Y%m%d_%H%M%S")
    return current_datetime_str

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_feature(df, feat, preprocess, train_mode=True):
    features = []
    labels = []
    
    for idx, row in tqdm(df.iterrows(), total = df.shape[0]):
        # librosa패키지를 사용하여 wav 파일 load
        y, sr = librosa.load('data'+row['path'][1:], sr=CONFIG.SR)
        # print(mfcc.shape)
        # librosa패키지를 사용하여 mfcc 추출
        if feat == 1:
            d = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CONFIG.N_MFCC)
        if feat == 2:
            d = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=CONFIG.n_fft, hop_length=CONFIG.hop_len, win_length=CONFIG.win_len, n_mels=CONFIG.n_mels)
        if feat == 0:
            d = pad_audio(y, CONFIG.SR * 5)
        if preprocess == 'None':
            pass
        if preprocess == 'pad':
            d = preprocess_spectrogram(d, max_length=CONFIG.max_len)
        if preprocess == 'mean':
            d = np.mean(d.T, axis=0)
        features.append(d)
       
        if train_mode:
            label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
            label_vector[0] = row['fake']
            label_vector[1] = row['real']
            labels.append(label_vector)

    if train_mode:
        return features, labels
    return features

def get_mstft_feature(df, train_mode=True):
    features = []
    labels = []
    d = []
    data = False
    if train_mode:
        if os.path.exists(f'data/train_mstft{str(CONFIG.n_mels)}.pickle'):
            with open(f'data/train_mstft{str(CONFIG.n_mels)}.pickle', 'rb') as file:
                data = pickle.load(file)
                print(f"Data loaded from data/train_mstft{str(CONFIG.n_mels)}.pickle")
    else:
        if os.path.exists(f'data/test_mstft{str(CONFIG.n_mels)}.pickle'):
            with open(f'data/test_mstft{str(CONFIG.n_mels)}.pickle', 'rb') as file:
                data = pickle.load(file)
                print(f"Data loaded from data/test_mstft{str(CONFIG.n_mels)}.pickle")
    for idx, row in tqdm(df.iterrows(), total = df.shape[0]):
        if data:
            mstft = data[idx]
        else:
            # librosa패키지를 사용하여 wav 파일 load
            y, sr = librosa.load('data/'+row['path'][1:], sr=CONFIG.SR)
            
            # librosa패키지를 사용하여 mstft 추출
            mstft = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=CONFIG.n_fft, hop_length=CONFIG.hop_len, win_length=CONFIG.win_len, n_mels=CONFIG.n_mels)
            d.append(mstft)
        if CONFIG.model == 'LCNN':
            mstft = preprocess_spectrogram(mstft, max_length=CONFIG.max_len)
        else:
            mstft = np.mean(mstft.T, axis=0)
        features.append(mstft)
        # if len(d):
        #     with open(f'data/train_mstft{str(CONFIG.n_mels)}.pickle', 'wb') as file:
        #             pickle.dump(d, file)
        #             print(f"Data saved as data/train_mstft{str(CONFIG.n_mels)}.pickle")

        if train_mode:
            label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
            label_vector[0] = row['fake']
            label_vector[1] = row['real']
            labels.append(label_vector)

    if train_mode:
        return features, labels
    return features

def multiLabel_AUC(y_true, y_scores):
    auc_scores = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        auc_scores.append(auc)
    mean_auc_score = np.mean(auc_scores)
    return mean_auc_score
    
def validation(model, main_criterion, val_loader, device, cent_criterion=None):
    model.eval()
    val_loss, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for features, labels in tqdm(iter(val_loader), desc='Validation'):
            features = features.float().to(device)
            labels = labels.float().to(device)
            
            f, probs = model(features)
            
            
            # cent_loss = cent_criterion(f, labels)
            # cent_loss *= CONFIG.cent_loss_weight
            loss = main_criterion(probs, labels)

            outputs_fp32 = probs.float()
            probs = torch.sigmoid(outputs_fp32)
            val_loss.append(loss.item())

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        _val_loss = np.mean(val_loss)

        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)

        # Calculate AUC score
        auc_score, brier_score, ece_score, combined_score = auc_brier_ece(all_labels, all_probs)

    
    return _val_loss, auc_score, brier_score, ece_score, combined_score

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
        x = x.to(torch.float16)
        distmat = distmat.to(torch.float16)
        distmat.addmm_(1, -2, x, self.centers.t().to(torch.float16))
        # print(distmat)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        # labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        # print(dist)
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
    
def check_data(cfg):
    feat = {0: 'raw', 1: 'mfcc_feat', 2:'mstft_feat'}
    train_d = cfg['train_data']
    f = cfg['feat']
    sr = cfg['sr']
    g = list(cfg[feat[f]].values())[0]
    p = cfg['preprocess']
    data_name = f'{train_d}_{f}_{g}_{sr}_{p}.sav'
    if os.path.exists(data_name):
        print(data_name+' found')
        data = joblib.load(data_name)
        # with open(data_name, 'rb') as file:
        #     data = pickle.load(file)
        #     print(f"Data loaded from {data_name}")
        return data
    else:
        return False, data_name

def pad_audio(audio, max_len):
    # print(audio.shape[0])
    if audio.shape[0] > max_len:
        return audio[:max_len]
    else:
        pad_width = (0, max_len - audio.shape[0])
        audio = np.pad(audio, pad_width, mode='constant')
        # print(audio.shape)
    return audio