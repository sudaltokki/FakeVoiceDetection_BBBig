import numpy as np
from tqdm import tqdm
import librosa
import pickle
import os

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


def get_mfcc_feature(df, args, train_mode=True):
    features = []
    labels = []
    d = []
    data = False
    
    for idx, row in tqdm(df.iterrows(), total = df.shape[0]):
        if data:
            mfcc = data[idx]
        else:
        # librosa패키지를 사용하여 wav 파일 load
            y, sr = librosa.load('data'+row['path'][1:], sr=args.sr)
        
        # librosa패키지를 사용하여 mfcc 추출
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=args.n_mfcc)
            d.append(mfcc)
        if args.classifier == 'LCNN':
            mfcc = preprocess_spectrogram(mfcc, max_length=args.max_len)
        else:
            mfcc = np.mean(mfcc.T, axis=0)
        features.append(mfcc)
       
        if train_mode:
            label_vector = np.zeros(args.n_classes, dtype=float)
            label_vector[0] = row['fake']
            label_vector[1] = row['real']
            labels.append(label_vector)

    if train_mode:
        return features, labels
    return features

def get_mstft_feature(df, args, train_mode=True):
    features = []
    labels = []
    d = []
    data = False
    if train_mode:
        if os.path.exists(f'data/train_mstft{str(args.n_mels)}.pickle'):
            with open(f'data/train_mstft{str(args.n_mels)}.pickle', 'rb') as file:
                data = pickle.load(file)
                print(f"Data loaded from data/train_mstft{str(args.n_mels)}.pickle")
    else:
        if os.path.exists(f'data/test_mstft{str(args.n_mels)}.pickle'):
            with open(f'data/test_mstft{str(args.n_mels)}.pickle', 'rb') as file:
                data = pickle.load(file)
                print(f"Data loaded from data/test_mstft{str(args.n_mels)}.pickle")
    for idx, row in tqdm(df.iterrows(), total = df.shape[0]):
        if data:
            mstft = data[idx]
        else:
            # librosa패키지를 사용하여 wav 파일 load
            y, sr = librosa.load('data'+row['path'][1:], sr=args.sr)
            
            # librosa패키지를 사용하여 mstft 추출
            mstft = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=args.n_fft, hop_length=args.hop_len, win_length=args.win_len, n_mels=args.n_mels)
            d.append(mstft)
        if args.classifier == 'LCNN':
            mstft = preprocess_spectrogram(mstft, max_length=args.max_len)
        else:
            mstft = np.mean(mstft.T, axis=0)
        features.append(mstft)
        # if len(d):
        #     with open(f'data/train_mstft{str(args.n_mels)}.pickle', 'wb') as file:
        #             pickle.dump(d, file)
        #             print(f"Data saved as data/train_mstft{str(args.n_mels)}.pickle")

        if train_mode:
            label_vector = np.zeros(args.n_classes, dtype=float)
            label_vector[0] = row['fake']
            label_vector[1] = row['real']
            labels.append(label_vector)

    if train_mode:
        return features, labels
    return features