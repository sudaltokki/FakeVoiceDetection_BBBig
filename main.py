from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import gc
import wandb
import joblib
from transformers import Wav2Vec2FeatureExtractor

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from config import CONFIG, wandb_config
from utils import *
from dataset import CustomDataset, Dataset_Train, Dataset_Eval, make_label
from models.model import *
from models.model_RawNet2 import *
from models.RawNet3 import *
from models.AASIST import AASIST

def train(model, optimizer, train_loader, val_loader, device, path):
    accumulation_step = int(CONFIG.TOTAL_BATCH_SIZE / CONFIG.BATCH_SIZE)
    model.to(device)
    main_criterion = nn.BCEWithLogitsLoss().to(device)

    best_val_score = 1
    best_model = None
    scaler = GradScaler()
   
    for epoch in tqdm(range(1, CONFIG.N_EPOCHS+1), desc='Train Epoch'):
        model.train()
        optimizer.zero_grad()
        train_loss = []

        for idx, (features, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training'):
            features = features.float().to(device)
            labels = labels.float().to(device)
            # print(features.shape)
            
            
            with autocast():
                if CONFIG.model == 'RNET2':
                    features, output = model(features, label=labels)
                else:
                    features, output = model(features)

                main_loss = main_criterion(output, labels)
                loss = main_loss / accumulation_step     

            scaler.scale(loss).backward()

            if (idx+1) % accumulation_step == 0 or (idx + 1) == len(train_loader):   
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss.append(main_loss.item())

        _val_loss, _auc, _brier, _ece, _val_score = validation(model, main_criterion, val_loader, device)
        _train_loss = np.mean(train_loss)

        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val Combined : [{_val_score:.5f}]')
        wandb.log({'Train_loss' : _train_loss, 'Val_loss' : _val_loss, 'Val_auc' : _auc, 'Val_brier' : _brier, 'Val_ece': _ece, 'Val_score': _val_score})
            
        # if best_val_score > _val_score:
        #     best_val_score = _val_score
        #     best_model = model
        torch.save(model.state_dict(), f'{path}/ep_{epoch}.pt')
    
    return best_model

def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for features in tqdm(iter(test_loader)):
            features = features.float().to(device)
            # print(features.shape)

            if CONFIG.model == 'RNET2':
                probs = model(features, is_test=True)
            else:
                f, probs = model(features)
            probs = torch.sigmoid(probs)

            if probs.is_cuda:
                probs = probs.cpu().detach().numpy()
            else:
                probs = probs.detach().numpy()
            predictions += probs.tolist()
    return predictions

def main():
    infer_model = None
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    # dummy_tensor = torch.cuda.FloatTensor(256, 1024, 1024)
    # dummy_tensor.data.resize_(1 * 1024**3)
    # print(torch.cuda.memory_summary())

    seed_everything(CONFIG.SEED) # Seed 고정
    date = get_date() 

    if CONFIG.feat == 1:
        input_dim = CONFIG.N_MFCC
    if CONFIG.feat == 2:
        input_dim = CONFIG.n_mels

    if CONFIG.feature_extractor:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(CONFIG.feature_extractor)
    else: 
        feature_extractor = CONFIG.feature_extractor
    print(f"feature extractor is {CONFIG.feature_extractor}")

    if CONFIG.model == 'MLP':
        model = MLP(input_dim, CONFIG.N_CLASSES)
    if CONFIG.model == 'LCNN':
        model = LCNN(input_dim, CONFIG.N_CLASSES)
    if CONFIG.model == 'RNET3':
        model = RawNet3(model_scale=8,
                        context=True,
                        summed=True,
                        encoder_type="ASP",
                        nOut=2,
                        out_bn=False,
                        sinc_stride=10,
                        log_sinc=True,
                        norm_sinc="mean",
                        grad_mult=1,
                        )
    if CONFIG.model == 'RNET2':
        model = RawNet2(
                RawNetBasicBlock,
                layers=[1, 1, 1, 2, 1, 2],
                nb_filters=[128, 128, 128, 256, 256, 256],
                nb_spk=2,
                code_dim=512,
            )
    if CONFIG.model == 'AASIST':
        model = AASIST(CONFIG.aasist_args)
    if CONFIG.model == 'ResNet18':
        model = ResNet18Audio()
    
    if CONFIG.finetune:
        model.load_state_dict(
            torch.load(CONFIG.finetune_model, map_location=device))
        print("Model loaded : {}".format(CONFIG.finetune_model))


    if CONFIG.train:
        wandb.init(
        # set the wandb project where this run will be logged
        project="antispoof",
        name = f'{date}_{CONFIG.user}',
        # track hyperparameters and run metadata
        config= wandb_config
        )
        
        df = pd.read_csv(CONFIG.TRAIN_PATH)
        # df = df[:1000]
        train_df, val_df, _, _ = train_test_split(df, df[['fake', 'real']], test_size=CONFIG.TEST_SIZE, random_state=CONFIG.SEED)
        # val_df.to_csv("val_df.csv", index=False)

        if CONFIG.model:
            train_feat, train_labels = make_label(train_df)
            val_feat, val_labels = make_label(val_df)
            train_dataset = Dataset_Train(train_feat, train_labels, feature_extractor, filter=CONFIG.fir_filter)
            val_dataset = Dataset_Train(val_feat, val_labels, feature_extractor, filter=CONFIG.fir_filter)
        
        else:
            data_exists = check_data(wandb_config)
            if data_exists[0] == False:
                train_feat, train_labels = get_feature(train_df, CONFIG.feat, CONFIG.mode, True)
                val_feat, val_labels = get_feature(val_df, CONFIG.feat, CONFIG.mode, True)

                feat_data = [train_feat, train_labels, val_feat, val_labels]
                joblib.dump(feat_data, data_exists[1])
                # with open(data_exists[1], "wb") as f:
                #     pickle.dump(feat_data, f)
                
            else:
                train_feat, train_labels, val_feat, val_labels = data_exists[0], data_exists[1], data_exists[2], data_exists[3]


            train_dataset = CustomDataset(train_feat, train_labels)
            val_dataset = CustomDataset(val_feat, val_labels)

            del data_exists
            gc.collect()

        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG.BATCH_SIZE,
            shuffle=True,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG.BATCH_SIZE,
            shuffle=False,
            pin_memory=True,
        )
        
        optimizer = torch.optim.Adam(params = model.parameters(), lr = CONFIG.LR)
        folder = f'ckpt/{date}'
        os.makedirs(folder, exist_ok=True)
        infer_model = train(model, optimizer, train_loader, val_loader, device, folder)
        print("MODEL READY!")

    if CONFIG.infer:
        if not CONFIG.train:
            model.load_state_dict(torch.load(CONFIG.infer_model))
            infer_model = model
            # torch.save(infer_model.state_dict(), 'ep_4_best.pt')
            print(f'{CONFIG.infer_model} successfully loaded!')

        # test_df = pd.read_csv('data/test.csv')
        #test_df = pd.read_csv('data/val.csv')
        # test_df = test_df[:100]
        if CONFIG.model:
            test_dataset = Dataset_Eval(test_df['path'].to_list(), feature_extractor, CONFIG.fir_filter)
        else:
            if os.path.exists('data/raw_test.sav'):
                test_feat = joblib.load('data/raw_test.sav')
            else:
                test_feat = get_feature(test_df, CONFIG.feat, CONFIG.mode, False)
                joblib.dump(test_feat, 'data/raw_test.sav')
            test_dataset = CustomDataset(test_feat, None)
        print('dataset ready!')
        test_loader = DataLoader(
            test_dataset,
            batch_size=CONFIG.BATCH_SIZE,
            shuffle=False
        )

        preds = inference(infer_model, test_loader, device)

        submit = pd.read_csv('data/sample_submission.csv')
        #submit = pd.read_csv('data/val_submission.csv')
        submit.iloc[:, 1:] = preds
        print(submit.head())

        submit.to_csv(f'result/{date}_{CONFIG.model}.csv', index=False)


if __name__ == "__main__":
    main()

    