from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, index):
        if self.label is not None:
            return self.feature[index], self.label[index]
        return self.feature[index]
