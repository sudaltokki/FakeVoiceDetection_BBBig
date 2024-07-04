from torch.utils.data import Dataset

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

class MSTFTDataset(Dataset):
    def __init__(self, mstft, label):
        self.mstft = mstft
        self.label = label

    def __len__(self):
        return len(self.mstft)

    def __getitem__(self, index):
        if self.label is not None:
            return self.mstft[index], self.label[index]
        return self.mstft[index]