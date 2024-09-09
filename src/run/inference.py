import torch
from torch import nn
from tqdm import tqdm


def inference(model, test_loader, device):
    sigmoid = nn.Sigmoid()
    
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for features in tqdm(iter(test_loader)):
            features = features.float().to(device)

            logits = model(features)
            probs = sigmoid(logits)

            probs  = probs.cpu().detach().numpy()
            predictions += probs.tolist()
    return predictions