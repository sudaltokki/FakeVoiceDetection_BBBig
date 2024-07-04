import pandas as pd
from sklearn.model_selection import train_test_split
import librosa
from config import CONFIG
from tqdm import tqdm
import numpy as np
import pickle
from model import LCNN
import torch

model = LCNN(16, 2)
model(torch.randn(96, 16, 256))
# x=torch.rand(96,16,32)
# batch_size = x.size(0)
# distmat = torch.pow(x, 2)
# print(distmat.shape)
# distmat = distmat.sum(dim=-1, keepdim=True)
# print(distmat.shape)
# distmat = distmat.expand(96, 2)
# print(distmat.shape)


# def get_mfcc_feature(df, train_mode=True):
#     features = []
#     labels = []
    
#     for _, row in tqdm(df.iterrows(), total = df.shape[0]):
#         # librosa패키지를 사용하여 wav 파일 load
#         y, sr = librosa.load('data/'+row['path'][1:], sr=CONFIG.SR)
        
#         # librosa패키지를 사용하여 mfcc 추출
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CONFIG.N_MFCC)
#         features.append(mfcc)

#         if train_mode:
#             label = row['label']
#             label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
#             label_vector[0 if label == 'fake' else 1] = 1
#             labels.append(label_vector)

#     if train_mode:
#         return features, labels
#     return features

# df = pd.read_csv('data/test.csv')
# train_feat = get_mfcc_feature(df, False)

# with open('data/test_mfcc16.pickle', 'wb') as f:
#     pickle.dump(train_feat, f)

# print("SUCCESSFULLY SAVED!")

with open('data/test_mstft.pickle', 'rb') as f:
    d = pickle.load(f)

print(d[0].shape)
length = []
for i in d:
    length.append(i[1])

df = pd.DataFrame(length)
print(df.describe())