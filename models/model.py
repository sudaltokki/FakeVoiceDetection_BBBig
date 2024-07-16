from torch import nn
import torch.nn.functional as F
import torch
import sys
from torchvision.models import resnet18

# ResNet18 모델 정의
class ResNet18Audio(nn.Module):
    def __init__(self):
        super(ResNet18Audio, self).__init__()
        self.resnet = resnet18(pretrained=False)
        
        # 첫 번째 합성곱 층의 입력 채널을 1로 수정 (오디오 특징 입력에 맞춤)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # 마지막 완전 연결 층의 출력을 2로 수정 (각 클래스에 대한 확률 출력)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)
        #self.sigmoid = nn.Sigmoid()  # Sigmoid 활성화 함수 추가
    
    def forward(self, features):
        features = features.unsqueeze(1).unsqueeze(3)
        x = self.resnet(features)
        return features, x
         # return self.sigmoid(x)  # Sigmoid 적용

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        y = self.fc3(x)
        # y = torch.sigmoid(y)
        return y
    
class LCNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LCNN, self).__init__()

        # LFCC dim
        self.feat_dim = in_dim
        
        # Define the model
        self.m_transform = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 64, [5, 5], 1, padding=[2, 2]),
                MaxFeatureMap2D(),
                nn.MaxPool2d([2, 2], [2, 2]),

                nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
                MaxFeatureMap2D(),
                nn.BatchNorm2d(32, affine=False),
                nn.Conv2d(32, 96, [3, 3], 1, padding=[1, 1]),
                MaxFeatureMap2D(),

                nn.MaxPool2d([2, 2], [2, 2]),
                nn.BatchNorm2d(48, affine=False),

                nn.Conv2d(48, 96, [1, 1], 1, padding=[0, 0]),
                MaxFeatureMap2D(),
                nn.BatchNorm2d(48, affine=False),
                nn.Conv2d(48, 128, [3, 3], 1, padding=[1, 1]),
                MaxFeatureMap2D(),

                nn.MaxPool2d([2, 2], [2, 2]),

                nn.Conv2d(64, 128, [1, 1], 1, padding=[0, 0]),
                MaxFeatureMap2D(),
                nn.BatchNorm2d(64, affine=False),
                nn.Conv2d(64, 64, [3, 3], 1, padding=[1, 1]),
                MaxFeatureMap2D(),
                nn.BatchNorm2d(32, affine=False),

                nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
                MaxFeatureMap2D(),
                nn.BatchNorm2d(32, affine=False),
                nn.Conv2d(32, 64, [3, 3], 1, padding=[1, 1]),
                MaxFeatureMap2D(),
                nn.MaxPool2d([2, 2], [2, 2]),
                
                nn.Dropout(0.7)
            )
        ])

        self.m_before_pooling = nn.ModuleList([
            nn.Sequential(
                BLSTMLayer((self.feat_dim // 16) * 32, (self.feat_dim // 16) * 32),
                BLSTMLayer((self.feat_dim // 16) * 32, (self.feat_dim // 16) * 32)
            )
        ])

        self.v_emd_dim = out_dim  # 2차원 출력으로 변경
        self.m_output_act = nn.ModuleList([
            nn.Linear((self.feat_dim // 16) * 32, self.v_emd_dim)
        ])

    def forward(self, x):
        # x: (batch_size, 1, n_lfcc, time)
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        #print(x.shape)

        for idx in range(len(self.m_transform)):
            # print(x.shape)
            x_sp_amp = self.m_transform[idx](x)
            #print(x_sp_amp.shape)
            x_sp_amp = x_sp_amp.permute(0, 3, 1, 2).contiguous()
            #print(x_sp_amp.shape)
            frame_num = x_sp_amp.shape[1]
            x_sp_amp = x_sp_amp.view(batch_size, frame_num, -1)
            #print(x_sp_amp.shape)
            x_lstm = self.m_before_pooling[idx](x_sp_amp)
            #print(x_lstm.shape)
            #print(x_lstm[0])
            y = torch.pow(x_lstm, 2).sum(dim=1)
            #print(y.shape)
            #print(y[0])
            tmp_emb = self.m_output_act[idx]((x_lstm + x_sp_amp).mean(1))
            #print(tmp_emb.shape)
        return y, torch.sigmoid(tmp_emb).squeeze(1)

class BLSTMLayer(torch.nn.Module):
    """ Wrapper over dilated conv1D
    Input tensor:  (batchsize=1, length, dim_in)
    Output tensor: (batchsize=1, length, dim_out)
    We want to keep the length the same
    """
    def __init__(self, input_dim, output_dim):
        super(BLSTMLayer, self).__init__()
        if output_dim % 2 != 0:
            print("Output_dim of BLSTMLayer is {:d}".format(output_dim))
            print("BLSTMLayer expects a layer size of even number")
            sys.exit(1)
        # bi-directional LSTM
        self.l_blstm = torch.nn.LSTM(input_dim, output_dim // 2, \
                                     bidirectional=True)
    def forward(self, x):
        # permute to (length, batchsize=1, dim)
        blstm_data, _ = self.l_blstm(x.permute(1, 0, 2))
        # permute it backt to (batchsize=1, length, dim)
        return blstm_data.permute(1, 0, 2)

class MaxFeatureMap2D(torch.nn.Module):
    """ Max feature map (along 2D) 
    
    MaxFeatureMap2D(max_dim=1)
    
    l_conv2d = MaxFeatureMap2D(1)
    data_in = torch.rand([1, 4, 5, 5])
    data_out = l_conv2d(data_in)

    
    Input:
    ------
    data_in: tensor of shape (batch, channel, ...)
    
    Output:
    -------
    data_out: tensor of shape (batch, channel//2, ...)
    
    Note
    ----
    By default, Max-feature-map is on channel dimension,
    and maxout is used on (channel ...)
    """
    def __init__(self, max_dim = 1):
        super().__init__()
        self.max_dim = max_dim
        
    def forward(self, inputs):
        # suppose inputs (batchsize, channel, length, dim)
        
        shape = list(inputs.size())
        
        if self.max_dim >= len(shape):
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But input has %d dimensions" % (len(shape)))
            sys.exit(1)
        if shape[self.max_dim] // 2 * 2 != shape[self.max_dim]:
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But this dimension has an odd number of data")
            sys.exit(1)
        shape[self.max_dim] = shape[self.max_dim]//2
        shape.insert(self.max_dim, 2)
        
        # view to (batchsize, 2, channel//2, ...)
        # maximize on the 2nd dim
        m, i = inputs.view(*shape).max(self.max_dim)
        return m
    
class Conv1dKeepLength(torch.nn.Conv1d):
    """ Wrapper for causal convolution
    Input tensor:  (batchsize=1, length, dim_in)
    Output tensor: (batchsize=1, length, dim_out)
    https://github.com/pytorch/pytorch/issues/1333
    Note: Tanh is applied
    """
    def __init__(self, input_dim, output_dim, dilation_s, kernel_s, 
                 causal = False, stride = 1, groups=1, bias=True, \
                 tanh = True, pad_mode='constant'):
        super(Conv1dKeepLength, self).__init__(
            input_dim, output_dim, kernel_s, stride=stride,
            padding = 0, dilation = dilation_s, groups=groups, bias=bias)
        
        self.causal = causal
        # input & output length will be the same        
        if self.causal:
            # left pad to make the convolution causal
            self.pad_le = dilation_s * (kernel_s - 1)
            self.pad_ri = 0
        else:
            # pad on both sizes
            self.pad_le = dilation_s * (kernel_s - 1) // 2
            self.pad_ri = dilation_s * (kernel_s - 1) - self.pad_le

        # we may wrap other functions too
        if tanh:
            self.l_ac = torch.nn.Tanh()
        else:
            self.l_ac = torch.nn.Identity()

        self.pad_mode = pad_mode
        #
        return

    def forward(self, data):
        # permute to (batchsize=1, dim, length)
        # add one dimension (batchsize=1, dim, ADDED_DIM, length)
        # pad to ADDED_DIM
        # squeeze and return to (batchsize=1, dim, length)
        # https://github.com/pytorch/pytorch/issues/1333
        x = torch.nn_func.pad(data.permute(0, 2, 1).unsqueeze(2), \
                              (self.pad_le, self.pad_ri, 0, 0),
                              mode = self.pad_mode).squeeze(2)
        # tanh(conv1())
        # permmute back to (batchsize=1, length, dim)
        output = self.l_ac(super(Conv1dKeepLength, self).forward(x))
        return output.permute(0, 2, 1)

class TimeInvFIRFilter(Conv1dKeepLength):                                    
    """ Wrapper to define a FIR filter over Conv1d
        Note: FIR Filtering is conducted on each dimension (channel)
        independently: groups=channel_num in conv1d
    """                                                                   
    def __init__(self, feature_dim, filter_coef, 
                 causal=True, flag_train=False):
        """ __init__(self, feature_dim, filter_coef, 
                 causal=True, flag_train=False)
        feature_dim: dimension of input data
        filter_coef: 1-D tensor of filter coefficients
        causal: FIR is causal or not (default: true)
        flag_train: whether train the filter coefficients (default: false)

        Input data: (batchsize=1, length, feature_dim)
        Output data: (batchsize=1, length, feature_dim)
        """
        super(TimeInvFIRFilter, self).__init__(                              
            feature_dim, feature_dim, 1, filter_coef.shape[0], causal,
            groups=feature_dim, bias=False, tanh=False)
        
        if filter_coef.ndim == 1:
            # initialize weight using provided filter_coef
            with torch.no_grad():
                tmp_coef = torch.zeros([feature_dim, 1, 
                                        filter_coef.shape[0]])
                tmp_coef[:, 0, :] = filter_coef
                tmp_coef = torch.flip(tmp_coef, dims=[2])
                self.weight = torch.nn.Parameter(tmp_coef, 
                                                 requires_grad=flag_train)
        else:
            print("TimeInvFIRFilter expects filter_coef to be 1-D tensor")
            print("Please implement the code in __init__ if necessary")
            sys.exit(1)

    def forward(self, data):                                              
        return super(TimeInvFIRFilter, self).forward(data)