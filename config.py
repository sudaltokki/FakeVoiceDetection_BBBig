class Config:
    # base
    user = 'bbbig' # 로그용인데 본인 이름 약자 쓰면 파일 저장 할때 쓰임
    model = 'MLP' # LCNN, MLP
    SR = 32000
    feat = 1 # [1:'MFCC', 2:'MSTFT']

    # mfcc
    N_MFCC = 16 # LCNN으로 돌릴 시 16이상 2의 배수로 설정
    # mstft
    n_fft=2048
    hop_len = 128
    win_len = 512
    n_mels = 60
    # Dataset
    TRAIN_PATH = 'data/org_com_not_sliced_noise_wav_now.csv'
    ROOT_FOLDER = './'
    # Preprocess
    max_len = 256

    # Training
    N_CLASSES = 2
    BATCH_SIZE = 96
    N_EPOCHS = 5
    LR = 3e-4
    TEST_SIZE = 0.2 # val data size
    cent_loss_weight = 0.5

    # Others
    SEED = 42
    train = True
    infer = True
    infer_model = 'ckpt/20240704_212832/ep_48_best.pt'
    
CONFIG = Config()

wandb_config= {
        "model": CONFIG.model,
        "sr": CONFIG.SR,
        "feat": CONFIG.feat,
        "batch": CONFIG.BATCH_SIZE,
        "epoch": CONFIG.N_EPOCHS,
        "lr": CONFIG.LR,
        "seed": CONFIG.SEED,
        "train_data" : CONFIG.TRAIN_PATH,
        "mfcc_feat" : {"n_mfcc":CONFIG.N_MFCC},
        "mstft_feat" : {"n_mels":CONFIG.n_mels, "n_fft":CONFIG.n_fft, "hop_len":CONFIG.hop_len, "win_len":CONFIG.win_len},

            }