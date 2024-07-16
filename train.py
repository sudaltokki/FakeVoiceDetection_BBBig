import torch
import torch.nn as nn
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor, TrainingArguments, Trainer
import evaluate
from datasets import DatasetDict, Dataset


metric = evaluate.load("accuracy")

# CONFIG and MODEL SETUP
model_name = 'amiriparian/ExHuBERT'
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
sampling_rate = feature_extractor.sampling_rate
label2id = {"fake":0, "real":1}
id2label= {0:"fake", 1:"real"}
model = AutoModelForAudioClassification.from_pretrained(model_name, trust_remote_code=True,label2id=label2id, id2label=id2label,num_labels=2,
                                                        revision="b158d45ed8578432468f3ab8d46cbe5974380812")

# Freezing half of the encoder for further transfer learning
model.freeze_og_encoder()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def preprocess_function(df):
    audios = []
    labels = []
    for idx, row in tqdm(df.iterrows(), total = df.shape[0]):
        y, sr = librosa.load('data/'+row['path'][1:], sr=sampling_rate)
        audios.append(y)
        label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
        label_vector[0] = row['fake']
        label_vector[1] = row['real']
        labels.append(label_vector)
    train_data = {
    'label': labels,
    'audio': audios
        }
    train_dataset = Dataset.from_dict(train_data)
    return train_dataset

def to_dataset(examples):
    audio_arrays = examples['audio']
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate),
        truncation=True,
        padding = True,
        return_attention_mask=True,
    )
    return inputs



from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
import wandb

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from config import CONFIG, wandb_config
from utils import *
from dataset import CustomDataset
from model import *

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def train(model, optimizer, train_loader, val_loader, device, path):
    model.to(device)


    main_criterion = nn.BCELoss().to(device)
    if CONFIG.model == 'LCNN':
        if CONFIG.feat == 1:
            cent_criterion = CenterLoss(feat_dim=CONFIG.N_MFCC*2).to(device)
        if CONFIG.feat == 2:
            cent_criterion = CenterLoss(feat_dim=(CONFIG.n_mels//16*32)).to(device)
    if CONFIG.model == 'MLP':
            cent_criterion = CenterLoss(feat_dim=128).to(device)
    optimizer_centloss = torch.optim.SGD(cent_criterion.parameters(), lr=0.5)
    
    best_val_score = 1
    best_model = None
    
    for epoch in tqdm(range(1, CONFIG.N_EPOCHS+1), desc='Train Epoch'):
        model.train()
        train_loss = []
        center_loss = []
        xent_loss = []
        for features, labels in iter(train_loader):
            features = features.float().to(device)
            labels = labels.float().to(device)
            # print(features.shape)
            
            optimizer.zero_grad()
            
            features, output = model(features)
            # print(output.shape, labels.shape)
            main_loss = main_criterion(output, labels)
            cent_loss = cent_criterion(features, labels)

            cent_loss *= CONFIG.cent_loss_weight
            loss = main_loss + cent_loss
       
            optimizer_centloss.zero_grad()

            loss.backward()
            optimizer.step()
        
            for param in cent_criterion.parameters():
                param.grad.data *= (1. / CONFIG.cent_loss_weight)
            optimizer_centloss.step()
            
            train_loss.append(main_loss.item())
            xent_loss.append(loss.item())
            center_loss.append(cent_loss.item())
                    
        _val_loss, _auc, _brier, _ece, _val_score = validation(model, main_criterion, cent_criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        _center_loss = np.mean(center_loss)
        _xent_loss = np.mean(xent_loss)

        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Xent_loss : [{_xent_loss:.5f}] Center Loss : [{_center_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val Combined : [{_val_score:.5f}]')
        wandb.log({'Train_loss' : _train_loss, 'Xent_loss':_xent_loss, 'Center_loss':_center_loss, 'Val_loss' : _val_loss, 'Val_auc' : _auc, 'Val_brier' : _brier, 'Val_ece': _ece, 'Val_score': _val_score})
            
        if best_val_score > _val_score:
            best_val_score = _val_score
            best_model = model
            torch.save(model, f'{path}/ep_{epoch}_best.pt')
    
    return best_model

def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for features in tqdm(iter(test_loader)):
            features = features.float().to(device)
            
            _, probs = model(features)

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

    seed_everything(CONFIG.SEED) # Seed 고정
    date = get_date()   

    if CONFIG.train:
        # wandb.init(
        # # set the wandb project where this run will be logged
        # project="antispoof",
        # name = f'{date}_{CONFIG.user}',
        # # track hyperparameters and run metadata
        # config= wandb_config
        # )
        
        df = pd.read_csv(CONFIG.TRAIN_PATH)
        #df = df[:55438]
        df = df[:100]
        y = preprocess_function(df)
        data_encoded = y.map(
            to_dataset,
            remove_columns=['audio'],
            batched=True,
            batch_size=32,
            num_proc=1,
        )


        train_data = data_encoded.train_test_split(test_size=CONFIG.TEST_SIZE, seed=CONFIG.SEED, shuffle=True)
        


        training_args = TrainingArguments(
            output_dir="my_awesome_mind_model",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=8,
            num_train_epochs=30,
            warmup_ratio=0.1,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False,
            fp16=True,

        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data['train'],
            eval_dataset=train_data['test'],
            tokenizer=feature_extractor,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        trainer.evaluate()

        
        # folder = f'ckpt/{date}'
        # os.makedirs(folder, exist_ok=True)
        # print("MODEL READY!")

    if CONFIG.infer:
        if not infer_model:
            infer_model = torch.load(CONFIG.infer_model)
            print(f'{CONFIG.infer_model} successfully loaded!')

        test_df = pd.read_csv('data/test.csv')
        # test_df = test_df[:100]
        if CONFIG.feat == 1:
            test_feat = get_mfcc_feature(test_df, False)
        if CONFIG.feat == 2:
            test_feat = get_mstft_feature(test_df, False)
        test_dataset = CustomDataset(test_feat, None)
        print('dataset ready!')
        test_loader = DataLoader(
            test_dataset,
            batch_size=CONFIG.BATCH_SIZE,
            shuffle=False
        )

        preds = inference(infer_model, test_loader, device)

        submit = pd.read_csv('data/sample_submission.csv')
        submit.iloc[:, 1:] = preds
        print(submit.head())

        submit.to_csv(f'result/{date}.csv', index=False)


if __name__ == "__main__":
    main()



