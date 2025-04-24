import os
import sys
import pandas as pd
import numpy as np
import torch
import glob
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import random_projection
from torchvision.models.inception import InceptionOutputs
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import pickle
import torchsummary  
import gc
import copy
import random
from tqdm import tqdm
import time
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity
from fvcore.nn import FlopCountAnalysis
from torchvision.models.densenet import _DenseBlock, _DenseLayer, _Transition
import ALERT_setting as setting
import timm
from timm.models.layers import to_2tuple,trunc_normal_


class ConcatModel(nn.Module):
    def __init__(self, base_model_t, base_model_f, classifier, lorv):
        super(ConcatModel, self).__init__()
        self.base_model_t = base_model_t
        self.base_model_f = base_model_f
        self.classifier = classifier
        self.lorv = lorv
        if self.lorv == 'RNN':
            self.lstm = LSTM()
            self.hidden_size = self.lstm.hidden_size
            self.layer_size = self.lstm.layer_size
            self.features_size = None
        elif self.lorv == 'DeiT':
            self.deit = VisionTransformer_deit()
        elif self.lorv == 'ViT':
            self.vit = VisionTransformer()


    def forward(self, t, f):
        features_t = self.base_model_t(t)
        features_f = self.base_model_f(f)
        if self.lorv == 'xLSTM':
            features_t = self.base_model_t(t, None)
            features_f = self.base_model_f(f, None)

        
        if self.lorv == 'RNN':
            #features = features_f # For MVL
            features = torch.cat((features_t, features_f), dim=2) #[Batch_size, seq_length, input_size]
            self.features_size = features.size(2) # is input_size
            output = self.lstm(features.float()).to(setting.device)
            # print(output.shape) # [25, 512]
            output = self.classifier(output, None)
        elif self.lorv == 'DeiT':
            output = self.deit(features_t, features_f) 
            output = self.classifier(output, None)
        elif self.lorv == 'ViT':
            output = self.vit(features_t, features_f) 
            output = self.classifier(output, None)
            #output = self.classifier(features_t, features_f)
        else:
            #output = self.classifier(features_f, None) # For MVL
            output = self.classifier(features_t, features_f)
     
        return output

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()         
        self.num_classes = num_classes
        self.initialized = False
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.fc1 = nn.Linear(3, 2)
        self.fc2 = nn.Linear(2, 1)
        self.fc3 = nn.Linear(1, 1)
        self.output_size = None

    def forward(self, t, f):
        # Concatenate the features from both models
        
        if f != None:
            #t = t * self.alpha
            #f = f * self.beta
            #print(self.alpha)
            output = torch.cat((t, f), dim=1)
        else:
            output = t
        feature = output
        self.output_size = output.shape[1]
        
        if not self.initialized:
            self.fc1 = nn.Linear(self.output_size, self.output_size//2).to(setting.device)
            self.fc2 = nn.Linear(self.output_size//2, self.output_size//4).to(setting.device)
            self.fc3 = nn.Linear(self.output_size//4, self.num_classes).to(setting.device)
            self.initialized = True
        output = torch.relu(self.fc1(output))
        output = torch.relu(self.fc2(output))
        output = self.fc3(output)
        
        return feature, output

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)

    def forward(self, x):
        # Convolution 1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Convolution 2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Flatten the network
        x = x.view(x.size(0), -1)
        
        return x


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        if setting.pretrained == True:
            self.alexnet = models.alexnet(pretrained=True)
        else:
            self.alexnet = models.alexnet(pretrained=False)
            self.alexnet.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2), bias=False)
        # Remove the fully connected layer (classifier)
        self.alexnet.classifier = torch.nn.Identity()

    def forward(self, x):
        if x.shape[2] < 70:
            x = F.pad(x, (0, 0, int((70-x.shape[2])/2), int((70-x.shape[2])/2))) # For making proper input size
        if setting.pretrained == True:
            x = x.repeat(1, 3, 1, 1)
        x = self.alexnet(x)
        x = x.view(x.size(0), -1)
        return x

# Not used
class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        
        self.vgg19 = models.vgg19(pretrained=False)
        self.vgg19.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.vgg19.classifier = torch.nn.Identity()
        
        
    def forward(self, x):
        x = self.vgg19(x)
        x = x.view(x.size(0), -1)
        return x

class GoogLeNet3(nn.Module):
    def __init__(self):
        super(GoogLeNet3, self).__init__()
        if setting.pretrained == True:
            self.googlenet3 = models.inception_v3(pretrained=True, aux_logits=True)
            #self.googlenet3.Conv2d_1a_3x3.conv = torch.nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
            self.googlenet3.aux_logits = False
            self.googlenet3.AuxLogits = None
            
        else:
            self.googlenet3 = models.inception_v3(pretrained=False, aux_logits=False)
            self.googlenet3.Conv2d_1a_3x3.conv = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.googlenet3.fc = torch.nn.Identity()
            
    def forward(self, x):
        if x.shape[2] < 80:
            x = F.pad(x, (0, 0, int((80-x.shape[2])/2), int((80-x.shape[2])/2))) # For making proper input size
        if setting.pretrained == True:
            x = x.repeat(1, 3, 1, 1)
        x = self.googlenet3(x)      
        if isinstance(x, InceptionOutputs):
            x = x.logits
        x = x.view(x.size(0), -1)
        return x

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        # Load a standard pre-trained ResNet50 model
        if setting.pretrained == True:
            self.resnet = models.resnet50(pretrained=True)
        #self.resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3,3), bias=False)
        # Remove the fully connected layer (classifier)
        else:
            self.resnet = models.resnet50(pretrained=False)
            self.resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=0, bias=False)
        self.resnet.fc = torch.nn.Identity()
        
    def forward(self, x):
        if setting.pretrained == True:
            x = x.repeat(1, 3, 1, 1)
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        return x


class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()
        if setting.pretrained == True:
            self.densenet = timm.create_model('densenet121', pretrained=True)#models.resnet18(pretrained=True)
        else:
            self.features = nn.Sequential(
                nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(num_init_features),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            
            # Each dense block
            num_features = num_init_features
            for i, num_layers in enumerate(block_config):
                block = _DenseBlock(
                    num_layers=num_layers,
                    num_input_features=num_features,
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    drop_rate=drop_rate,
                )
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                
                if i != len(block_config) - 1:  # do not add transition layer after the last block 
                    trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2) # transition layer reduces the channel for calculation efficiency
                    self.features.add_module('transition%d' % (i + 1), trans)
                    num_features = num_features // 2
            
            # Final batch norm
            self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        
        self.densenet.classifier = torch.nn.Identity()
        
    def forward(self, x):
        if setting.pretrained == True:
            x = x.repeat(1, 3, 1, 1)
            x = self.densenet(x)
            x = x.view(x.size(0), -1)
        else:
            features = self.features(x)
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
        return x

class MobileNet3(nn.Module):
    def __init__(self):
        super(MobileNet3, self).__init__()
        if setting.pretrained == True:
            self.mobilenet3 = models.mobilenet_v3_large(pretrained=True)
        else:
            self.mobilenet3 = models.mobilenet_v3_large(pretrained=False)
            self.mobilenet3.features[0][0] = torch.nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding = (1, 1), bias=False)
        
        self.mobilenet3.classifier = torch.nn.Identity()
        
    def forward(self, x):
        if setting.pretrained == True:
            x = x.repeat(1, 3, 1, 1)
        x = self.mobilenet3(x)
        x = x.view(x.size(0), -1)
        return x

# Not used
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.hidden_size = 1024 #1024 in CNNRNN
        self.layer_size = 1 # 1in CNNRNN
        self.initialized = False
        self.input_size = 51 #don't care number
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.layer_size, batch_first=True, bidirectional=True) #True in CNNRNN
        #self.lstm = nn.RNN(self.input_size, self.hidden_size, batch_first=True)
        
    def forward(self, x):
        if not self.initialized:
            input_size = x.size(2)
            #print(input_size) #140
            self.lstm = nn.LSTM(input_size, self.hidden_size, self.layer_size, batch_first=True, bidirectional=True).to(setting.device)
            #self.lstm = nn.RNN(input_size, self.hidden_size).to(setting.device)
            self.initialized = True            
        h0 = torch.zeros(self.layer_size*2, x.size(0), self.hidden_size).to(setting.device)
        c0 = torch.zeros(self.layer_size*2, x.size(0), self.hidden_size).to(setting.device)
        out, _ = self.lstm(x, (h0,c0)) #[sequence length, batch_size, hidden size(output_size)]
        #out, _ = self.lstm(x, h0) #[sequence length, batch_size, hidden size(output_size)]
        
        out = out[:, -1, :]
        #print(out.shape)
        return out

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder1 = nn.Linear(input_dim, input_dim//2)
        self.encoder2 = nn.Linear(input_dim//2, input_dim//4)
        self.encoder3 = nn.Linear(input_dim//4, encoding_dim)
        # Decoder
        #self.decoder = nn.Linear(encoding_dim, input_dim)

    def forward(self, x):
        x = torch.relu(self.encoder1(x))
        x = torch.relu(self.encoder2(x))
        x = self.encoder3(x)
        #x = self.decoder(x)
        return x

class pre_LSTM(nn.Module):
    def __init__(self):
        super(pre_LSTM, self).__init__()
        self.feature_list = []
        self.input_dim = 50 * 51  # Assuming the size of each snippet after flattening
        self.encoding_dim = 256
        self.ae = Autoencoder(self.input_dim, self.encoding_dim).to(setting.device)
        self.initialized = False
    def forward(self, x):
        feature_list = []
        x = x.squeeze(1).transpose(1,2) #[batch, sequence, input_size]
        snippet  = int(x.size(1) / 10)
        if not self.initialized:
            self.ae = Autoencoder(snippet*x.size(2), 256).to(setting.device)
            self.initialized = True
        for i in range(10):
            snippet_feature = torch.flatten(x[:, snippet*i:snippet*(i+1), :], 1)
            snippet_feature = self.ae(snippet_feature)
            feature_list.append(snippet_feature)
        feature_list = torch.stack(feature_list, dim=1)
        
        return feature_list

class CNN_RNN(nn.Module):
    def __init__(self):
        super(CNN_RNN, self).__init__()
        self.mini_ResNet = ResNet50()
        self.feature_list = []

    def forward(self, x):
        feature_list = []
        snippet = int(x.size(3) / 10)
        for i in range(10):
            snippet_feature = self.mini_ResNet(x[:,:,:,snippet*i:snippet*(i+1)])
            feature_list.append(snippet_feature)
        #feature_list=torch.tensor(np.array(feature_list))
        #feature_list = feature_list.transpose(0,1)
        feature_list = torch.stack(feature_list, dim=1)
        return feature_list #[batch, snippet, input_size]


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_height=None, patch_width=50, emb_size=768):
        super().__init__()
        assert patch_height is not None, "Patch height must be defined."
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.projection = nn.Sequential(
            # Using a convolutional layer to break into patches of full height x 50 width
            nn.Conv2d(in_channels, emb_size, kernel_size=(patch_height, patch_width), stride=(patch_height, patch_width)),
            nn.Flatten(2)  # Flatten the patches
        )

    def forward(self, x):
        x = self.projection(x)  # [B, E, N] where N is the number of patches (10)
        
        return x

# Simple Vision Transformer Architecture
class pre_Transformer(nn.Module):
    def __init__(self, img_height=None, num_patches=10, emb_size=384):
        super().__init__()
        
    def forward(self, x):
        x = nn.Upsample(size=(224, 224), mode='bilinear')(x)
        return x


class VisionTransformer_deit(nn.Module):
    def __init__(self, img_height=None, num_patches=10, emb_size=384, num_heads=16, depth=16): #num_head:12, depth:12, emb_size:768
        super().__init__()
        if setting.pretrained == True:
            self.model = timm.create_model('deit_tiny_distilled_patch16_224', pretrained=True)
        else:
            self.model = timm.create_model('deit_tiny_distilled_patch16_224', pretrained=False)
        
        #self.model.patch_embed.proj = torch.nn.Conv2d(1, 192, kernel_size=(16, 16), stride=(16, 16)) 
        #self.model.patch_embed.proj = torch.nn.Conv2d(2, 192, kernel_size=(16, 16), stride=(16, 16))
        
        #weight change 
        self.model.head = torch.nn.Identity()#torch.nn.Linear(192, 192)
        self.model.head_dist = torch.nn.Identity()

    def forward(self, x1, x2):
        x = x1 # For MVL
        x = x.repeat(1, 3, 1, 1)# For MVL
        #x = torch.cat((x1, x2), dim=1) #[B, 2, 244, 244]
        #third_channel = x.mean(dim=1, keepdim=True)  # Calculate the mean of the two channels
        #x = torch.cat((x, third_channel), dim=1)
        x = self.model(x)
        return x  # Class prediction

class RaDA(nn.Module):
    def __init__(self):
        super(RaDA, self).__init__()
        self.mini_ResNet = ResNet50()
        self.feature_list = []

    def forward(self, x):
        feature_list = []
        snippet = int(x.size(3) / 5)
        for i in range(5):
            snippet_feature = self.mini_ResNet(x[:,:,:, snippet * i : snippet * (i + 1)])
            feature_list.append(snippet_feature)
        #feature_list=torch.tensor(np.array(feature_list))
        #feature_list = feature_list.transpose(0,1)
        #print(snippet_feature.shape)
        feature_list = torch.cat(feature_list, dim=1)
        #print(feature_list.shape)
        return feature_list 

class VisionTransformer(nn.Module):
    def __init__(self, img_height=None, num_patches=10, emb_size=384, num_heads=16, depth=16): #num_head:12, depth:12, emb_size:768
        super().__init__()
        if setting.pretrained == True:
            self.model = timm.create_model('vit_large_patch16_224', pretrained=True)
        else:
            print("We don't provide the scratch model.")
            sys.exit()
        #self.model.patch_embed.proj = torch.nn.Conv2d(1, 192, kernel_size=(16, 16), stride=(16, 16))
        #self.model.patch_embed.proj = torch.nn.Conv2d(2, 1024, kernel_size=(16, 16), stride=(16, 16))
        #weight change 
        self.model.head = torch.nn.Identity()#torch.nn.Linear(192, 192)
        #self.model.head_dist = torch.nn.Identity()

    def forward(self, x1, x2):
        #x = x1 # For MVL
        #x = x.repeat(1, 3, 1, 1)# For MVL
        x = torch.cat((x1, x2), dim=1) #[B, 2, 244, 244]
        third_channel = x.mean(dim=1, keepdim=True)  # Calculate the mean of the two channels
        x = torch.cat((x, third_channel), dim=1)
        x = self.model(x)
        return x  # Class prediction        

class xLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Extended LSTM cell with an additional multiplicative integration term.
        This design follows the ideas of the original xLSTM (Beck et al. 2024).
        """
        super(xLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Standard LSTM gate parameters
        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=False)

        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)

        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=False)

        self.W_g = nn.Linear(input_size, hidden_size)
        self.U_g = nn.Linear(hidden_size, hidden_size, bias=False)

        # Extended multiplicative term parameters
        self.W_x = nn.Linear(input_size, hidden_size)
        self.U_x = nn.Linear(hidden_size, hidden_size, bias=False)

        # Gate for modulating the extended term
        self.W_a = nn.Linear(input_size, hidden_size)
        self.U_a = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h_prev, c_prev):
        # Standard LSTM computations
        i = torch.sigmoid(self.W_i(x) + self.U_i(h_prev))
        f = torch.sigmoid(self.W_f(x) + self.U_f(h_prev))
        o = torch.sigmoid(self.W_o(x) + self.U_o(h_prev))
        g = torch.tanh(self.W_g(x) + self.U_g(h_prev))
        
        # Extended multiplicative interaction term
        m = self.W_x(x) * self.U_x(h_prev)  # element-wise multiplication
        a = torch.sigmoid(self.W_a(x) + self.U_a(h_prev))
        extended_term = a * m

        # Update cell state and hidden state
        c_new = f * c_prev + i * g + extended_term
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

###############################################################################
# Multi-layer xLSTM Module
###############################################################################
class xLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0):
        """
        Multi-layer xLSTM module. It stacks xLSTMCells and applies dropout between layers.
        Args:
            input_size: Feature dimension per time step.
            hidden_size: Hidden state dimension.
            num_layers: Number of xLSTM layers.
            dropout: Dropout probability.
        """
        super(xLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.cells = nn.ModuleList([
            xLSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, hidden=None):
        # x: shape [seq_len, batch, input_size]
        seq_len, batch_size, _ = x.size()
        if hidden is None:
            h_list = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
            c_list = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        else:
            h_list, c_list = hidden

        outputs = []
        # Process each time step sequentially.
        for t in range(seq_len):
            input_t = x[t]  # shape: [batch, input_size]
            for layer in range(self.num_layers):
                h, c = self.cells[layer](input_t, h_list[layer], c_list[layer])
                h_list[layer] = h
                c_list[layer] = c
                input_t = h
                if self.dropout and layer < self.num_layers - 1:
                    input_t = self.dropout_layer(input_t)
            outputs.append(h_list[-1].unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (h_list, c_list)

###############################################################################
# Time Series Feature Extraction Model using xLSTM
###############################################################################
class XLSTMTimeSeriesFeatureModel(nn.Module):
    def __init__(self, input_feature_size, hidden_size, num_layers, dropout=0.5):
        """
        Time series model using xLSTM, adapted to return a feature vector at the last time step.
        This version replaces the decoder for classification with a feature extractor.
        Args:
            input_feature_size: Number of features per time step (51).
            hidden_size: Hidden dimension.
            num_layers: Number of xLSTM layers.
            dropout: Dropout probability.
        """
        super(XLSTMTimeSeriesFeatureModel, self).__init__()
        self.input_feature_size = input_feature_size  # Should be 51.
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input projection: project raw input features into hidden space.
        self.input_projection = nn.Linear(input_feature_size, hidden_size)
        # xLSTM module
        self.xlstm = xLSTM(hidden_size, hidden_size, num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.input_projection.bias.data.zero_()

    def forward(self, x, hidden=None):
        """
        Args:
            x: Input tensor of shape [batch, channel, height, width] where
               channel = 1, height = 51 (features), width = 500 (time steps).
            hidden: Initial hidden state (optional).
        Returns:
            feature: Tensor of shape [batch, hidden_size] representing the feature
                     at the final time step.
            hidden: Final hidden states.
        """
        # Remove the singleton channel dimension: [batch, 1, 51, 500] -> [batch, 51, 500]
        x = x.squeeze(1)
        # Rearrange to [batch, time, features]: [batch, 51, 500] -> [batch, 500, 51]
        x = x.permute(0, 2, 1)
        # Transpose to get [seq_len, batch, input_feature_size]: [500, batch, 51]
        x = x.transpose(0, 1)

        # Project input features from 51 to hidden_size dimensions.
        x = self.input_projection(x)
        # Process sequence through the xLSTM module.
        outputs, hidden = self.xlstm(x, hidden)
        outputs = self.dropout(outputs)
        # Extract feature: take the output at the final time step.
        # outputs has shape [seq_len, batch, hidden_size]; we take outputs[-1].
        feature = outputs[-1]  # shape: [batch, hidden_size]
        return feature#, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        h0 = [weight.new_zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        c0 = [weight.new_zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        return (h0, c0)


def model_selection(model):
    if model == 'LeNet': # LeNet-5
        base_model_t = LeNet5()
        base_model_f = LeNet5()
        epochs = 30
        lr = 0.3 # SGD:0.01~0.1, adam:0.0001~0.001
        criterion = nn.CrossEntropyLoss() 
    elif model == 'AlexNet':
        base_model_t = AlexNet()
        base_model_f = AlexNet()
        epochs = 30
        lr = 0.5 # SGD 0.001~0.01
        criterion = nn.CrossEntropyLoss()
    elif model == 'VGG':
        base_model_t = VGG19()
        base_model_f = VGG19()
        epochs = 30
        lr = 0.05 # Recommended 0.01 
        criterion = nn.CrossEntropyLoss()
    elif model == 'GoogLeNet':
        base_model_t = GoogLeNet3()
        base_model_f = GoogLeNet3()
        epochs = 30
        lr = 0.0001
        criterion = nn.CrossEntropyLoss()
    elif model == 'ResNet':
        base_model_t = ResNet50()
        base_model_f = ResNet50()
        epochs = 30
        lr = 0.0001
        criterion = nn.CrossEntropyLoss()
    elif model == 'DenseNet':
        base_model_t = DenseNet121()
        base_model_f = DenseNet121()
        epochs = 30
        lr = 0.0001
        criterion = nn.CrossEntropyLoss()
    elif model == 'MobileNet':
        base_model_t = MobileNet3()
        base_model_f = MobileNet3()
        epochs = 30
        lr = 0.001 # SGD:0.005~0.01, Adam:0.001
        criterion = nn.CrossEntropyLoss()
    elif model == 'RNN':
        base_model_t = pre_LSTM()
        base_model_f = pre_LSTM()
        epochs = 30
        lr = 0.0001 #adam:0.001, RMSprop:0.001
        criterion = nn.CrossEntropyLoss()
    elif model == 'CNN+RNN':
        base_model_t = CNN_RNN()
        base_model_f = CNN_RNN()
        epochs = 30
        lr = 0.0001
        criterion = nn.CrossEntropyLoss()
    elif model == 'DeiT':
        base_model_t = pre_Transformer()
        base_model_f = pre_Transformer()
        epochs = 30
        lr = 0.00001 #AdamW: 0.0001~0.001, optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
        criterion = nn.CrossEntropyLoss()
    elif model == 'RaDA':
        base_model_t = RaDA()
        base_model_f = RaDA()
        epochs = 30
        lr = 0.0001 
        criterion = nn.CrossEntropyLoss()
    elif model == 'ViT':
        base_model_t = pre_Transformer()
        base_model_f = pre_Transformer()
        epochs = 30
        lr = 0.00001 
        criterion = nn.CrossEntropyLoss()
    elif model == 'xLSTM':
        base_model_t = XLSTMTimeSeriesFeatureModel(51, 128, 2, dropout=0.5)
        base_model_f = XLSTMTimeSeriesFeatureModel(89, 128, 2, dropout=0.5)
        epochs = 30
        lr = 0.00001 
        criterion = nn.CrossEntropyLoss()
    elif model == 'all':
        for lr_alg in learning_algorithms:
            None
    else:
        pass
    
    classifier = Classifier(setting.num_classes)
    if model == 'RNN' or model == 'CNN+RNN':
        base_model = ConcatModel(base_model_t, base_model_f, classifier, 'RNN').to(setting.device)
    elif model == 'DeiT':
        base_model = ConcatModel(base_model_t, base_model_f, classifier, 'DeiT').to(setting.device)
    elif model == 'ViT':
        base_model = ConcatModel(base_model_t, base_model_f, classifier, 'ViT').to(setting.device)
    elif model == 'xLSTM':
        base_model = ConcatModel(base_model_t, base_model_f, classifier, 'xLSTM').to(setting.device)

    else:
        base_model = ConcatModel(base_model_t, base_model_f, classifier, None).to(setting.device)
    
    if model == 'LeNet': # LeNet-5
        optimizer = optim.SGD(base_model.parameters(), lr=lr)    
    elif model == 'AlexNet':
        optimizer = optim.SGD(base_model.parameters(), lr=lr)
    elif model == 'VGG':
        optimizer = optim.SGD(base_model.parameters(), lr=lr)
    elif model == 'GoogLeNet':
        optimizer = optim.Adam(base_model.parameters(), lr=lr)  # Recommended RMSprop
    elif model == 'ResNet':
        optimizer = optim.Adam(base_model.parameters(), lr=lr)
    elif model == 'DenseNet':
        optimizer = optim.Adam(base_model.parameters(), lr=lr)
    elif model == 'MobileNet':
        optimizer = optim.Adam(base_model.parameters(), lr=lr)
    elif model == 'RNN':
        optimizer = optim.Adam(base_model.parameters(), lr=lr)
    elif model == 'CNN+RNN':
        optimizer = optim.Adam(base_model.parameters(), lr=lr)
    elif model == 'DeiT':
        optimizer = optim.Adam(base_model.parameters(), lr=lr)#optim.AdamW(base_model.parameters(), lr=lr, weight_decay=0.05) #optim.SGD(base_model.parameters(), lr=lr, momentum=0.9)
    elif model == 'RaDA':
        optimizer = optim.Adam(base_model.parameters(), lr=lr)
    elif model == 'ViT':
        optimizer = optim.Adam(base_model.parameters(), lr=lr)#optim.AdamW(base_model.parameters(), lr=lr, weight_decay=0.05) #optim.SGD(base_model.parameters(), lr=lr, momentum=0.9)
    elif model == 'xLSTM':
        optimizer = optim.Adam(base_model.parameters(), lr=lr)
    elif model == 'all':
        pass
    else:
        pass
    
    return base_model, optimizer, criterion, epochs, lr
