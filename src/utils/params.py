import argparse

from datetime import datetime
import pytz

def get_datetime():
    tz = pytz.timezone('Asia/Seoul')
    current_time = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
    return current_time

current_time = get_datetime()


def parse_args(args):
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument("--feature_extractor", type=str, default='mfcc', help="mfcc / mstft")
    parser.add_argument("--classifier", type=str, default='MLP', help="MLP / LCNN")
    
    # preprocess
    parser.add_argument("--sr", type=int, default=16000, help="Sample Rate")
    parser.add_argument("--mel_max_len", type=int, default=60, help="mel spetrogram max length")
    # mfcc
    parser.add_argument("--n_mfcc", type=int, default=16, help="mfcc")
    # mstft
    parser.add_argument("--n_fft", type=int, default=2048, help="mstft")
    parser.add_argument("--hop_len", type=int, default=128, help="mstft")
    parser.add_argument("--win_len", type=int, default=512, help="mstft")
    parser.add_argument("--n_mels", type=int, default=60, help="mstft0")

    # training/validation
    parser.add_argument("--train", type=bool, default=True, help="train")
    parser.add_argument("--train_csv_path", type=str, default='data/noise_added_all/new_noise_added_all.csv', help="train data path")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="train test ratio")
    parser.add_argument("--n_classes", type=int, default=2, help="classes")   
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU.")
    parser.add_argument("--total_batch_size", type=int, default=64, help="Total batch size per GPU.")
    parser.add_argument("--epochs", type=int, default=5, help="epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--log_path", type=str, default="logs", help="log path")
    
    # infer
    parser.add_argument("--infer", type=bool, default=True, help="infer")
    parser.add_argument("--model_checkpoint", type=str, default="", help="load model checkpoint")
    parser.add_argument("--test_csv_path", type=str, default='data/test.csv', help="test data path")
    parser.add_argument("--submission_csv_path", type=str, default='data/sample_submission.csv', help="submission data path")

    # loss weight
    parser.add_argument("--cent_loss_weight", type=float, default=0.5, help="cent loss weight")
    
    # others
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--current_time", type=str, default=current_time, help="current time")

    # wandb
    parser.add_argument("--set_wandb", type=bool, default=True, help="wandb")
    parser.add_argument("--user", type=str, default="", help="user name")


    args = parser.parse_args(args)

    return args
