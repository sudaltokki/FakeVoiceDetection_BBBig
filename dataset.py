from torch.utils.data import Dataset
import numpy as np
from torch import Tensor
import torch
import librosa
from config import CONFIG
from scipy.signal import firwin, lfilter

class CustomDataset(Dataset):
    def __init__(self, mfcc, label):
        self.mfcc = mfcc
        self.label = label

    def __len__(self):
        return len(self.mfcc)

    def __getitem__(self, index):
        if self.label is not None:
            return self.mfcc[index], self.label[index]
        return self.mfcc[index]

class RNETDataset(Dataset):
    def __init__(self, audio, label):
        self.audio = audio
        self.label = label

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, index):
        if self.label is not None:
            return self.audio[index], self.label[index]
        return self.audio[index]
    
def pad(x, max_len=160000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def make_label(df):
    labels = []
    path = []
    for idx, row in df.iterrows():
        path.append(row['path'])
        label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
        label_vector[0] = row['fake']
        label_vector[1] = row['real']
        labels.append(label_vector)
    return path, labels

def pad_random(x: np.ndarray, max_len: int = 160000):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len > max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x

def filter(x, sr):
    bpf = firwin(101, [1000, 7000], pass_zero='bandpass', fs=sr)
    filtered_signal = lfilter(bpf, 1.0, x)
    return filtered_signal

class Dataset_Train(Dataset):
    def __init__(self, audio, labels, feature_extractor=False, filter=False):
        self.audio = audio
        self.labels = labels
        self.cut = 80000  # take ~5 sec audio (64600 samples)
        self.feature_extractor = feature_extractor
        self.filter = filter

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, index):
        X, sr = librosa.load('data'+self.audio[index][1:], sr=CONFIG.SR)
        X = pad_random(X, sr*5)
        if self.feature_extractor:
            X = self.feature_extractor(
                raw_speech=X,
                sampling_rate=CONFIG.SR,
                padding=False,
                max_length=5,
                return_tensors="pt"
            )
            X = X['input_values'][0]
        if self.filter:
            X = filter(X, sr)
        if not torch.is_tensor(X):
            X = torch.tensor(X)
        y = self.labels[index]
        y = Tensor(y)
        return X, y


class Dataset_Eval(Dataset):
    def __init__(self, audio, feature_extractor=False, filter=False):
        self.audio = audio
        self.cut = 80000  # take ~5 sec audio (64600 samples)
        self.feature_extractor = feature_extractor
        self.filter = filter

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, index):
        X, sr = librosa.load('data'+self.audio[index][1:], sr=CONFIG.SR)
        X = pad_random(X, sr*5)
        if self.feature_extractor:
            X = self.feature_extractor(
                raw_speech=X,
                sampling_rate=CONFIG.SR,
                padding=False,
                max_length=5,
                return_tensors="pt"
            )
            X = X['input_values'][0]
        if self.filter:
            X = filter(X, sr)
        if not torch.is_tensor(X):
            X = torch.tensor(X)
        return X