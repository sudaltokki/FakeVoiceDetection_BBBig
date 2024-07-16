class Config:
    # base
    user = 'PJY' # 로그용인데 본인 이름 약자 쓰면 파일 저장 할때 쓰임
    feature_extractor = 'facebook/hubert-base-ls960' #'facebook/wav2vec2-xls-r-300m' "facebook/hubert-base-ls960" 'facebook/hubert-large-ls960-ft' 'microsoft/wavlm-base-plus'
    model = 'AASIST' # LCNN, MLP, RNET2, AASIST
    SR = 16000
    feat = 0 # [0: 'raw', 1:'MFCC', 2:'MSTFT']

    # # mfcc
    # N_MFCC = 16 # LCNN으로 돌릴 시 16이상 2의 배수로 설정
    # # mstft
    # n_fft=2048
    # hop_len = 128
    # win_len = 512
    # n_mels = 60
    # # Dataset
    TRAIN_PATH = 'data/noise_added_all.csv'
    ROOT_FOLDER = './'
    # Preprocess
    mode = None #pad, mean
    max_len = 256
    fir_filter = False

    # Training
    N_CLASSES = 2


    TOTAL_BATCH_SIZE = 64
    BATCH_SIZE = 16
    N_EPOCHS = 20
    LR = 1e-5
    TEST_SIZE = 0.2 # val data size
    cent_loss_weight = 0.5
    # AASIST
    aasist_args = {"first_conv": 128,
              "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
              "gat_dims": [64, 32],
              "pool_ratios": [0.5, 0.7, 0.5, 0.5],
              "temperatures": [2.0, 2.0, 100.0, 100.0]
              }

    

    # Others
    SEED = 42
    train = True
    finetune = True
    finetune_model = 'models/weights/AASIST.pth'
    infer = True
    infer_model = 'ckpt/20240712_160149_hubert+ResNet18_ep_5_best.pt'
    
CONFIG = Config()

wandb_config= {
        "model": CONFIG.model,
        "feature_extractor" : CONFIG.feature_extractor,
        "sr": CONFIG.SR,
        "feat": CONFIG.feat,
        "total_batch": CONFIG.TOTAL_BATCH_SIZE,
        "batch": CONFIG.BATCH_SIZE,
        "epoch": CONFIG.N_EPOCHS,
        "lr": CONFIG.LR,
        "seed": CONFIG.SEED,
        "train_data" : CONFIG.TRAIN_PATH.replace('.csv', ''),
        # "preprocess" :  CONFIG.mode,
        # "mfcc_feat" : {"n_mfcc":CONFIG.N_MFCC},
        # "mstft_feat" : {"n_mels":CONFIG.n_mels, "n_fft":CONFIG.n_fft, "hop_len":CONFIG.hop_len, "win_len":CONFIG.win_len},
        "raw" : {"sec": 0},
        # "aasist" : CONFIG.aasist_args
            }