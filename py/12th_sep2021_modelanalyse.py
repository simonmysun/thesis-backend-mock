 #!/usr/bin/env python
print ("HANDLING IMPORTS...")
import sys
sys.path.append("..")
import os
import math
import time
import operator
import json
from threading import Thread
import copy
import cv2
import numpy as np
#import matplotlib.pyplot as plt
import pyaudio
import librosa
from pathlib import Path
import pandas as pd
from scipy import  misc
import config as cfg
import pandas as pd
import torch
from fastprogress import progress_bar
import warnings
from collections import defaultdict
from collections import Counter
import pymongo
from pymongo import MongoClient
from utils import log
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["mydatabase"]
import datetime

mycol = mydb["AALdanny94_edited_german_25thjan2021"]

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import uuid
import paho.mqtt.client as mqtt
def mqtt_on_connect(mqtt_client, userdata, flags, rc):
    print("[MQTT] Connected with result code "+str(rc))
def mqtt_on_message(mqtt_client, userdata, msg):
    print("[MQTT] received on \"%s\" : \"%s\" " % (msg.topic, str(msg.payload)))
mqtt_client_id = 'data_source_py_%s' % uuid.uuid4()
print("[MQTT] Initializing MQTT Client %s" % mqtt_client_id)
mqtt_topic = "tele/indoor_sound_classification/fake_datasource_2/state"
mqtt_client = mqtt.Client(mqtt_client_id)
mqtt_client.on_connect = mqtt_on_connect
mqtt_client.on_message = mqtt_on_message
mqtt_client.username_pw_set(username="TuCDataSource", password="b52393dda08af7a991e7af074f377997a082f652")
mqtt_client.connect("mqtt.makelove.expert", port=1883, keepalive=60)
mqtt_client.loop_start()
mqtt_time_last_send = time.time()
mqtt_msg_freq = 1 / 20
mqtt_debounce = list()

##################################################################################################################################################################################
AAL_CODE  = {
'AlaramClock':0, 
'Blending':1, 
'Breaking':2, 
'Canopening':3, 
'Cat':4, 
'Chirpingbirds':5, 
'Clapping':6, 
'Clarinet':7, 
'Clocktick':8, 
'Crying':9, 
'Electronic_toothbrush':10, 
'Displaying_furniture':11, 
'Dog':12, 
'DoorBell':13, 
'Dragonground':14, 
'Drill':15, 
'Drinking':16, 
'Drum':17, 
'Femalespeaking':18, 
'Flute':19, 
'Glass':20, 
'Guitar':21, 
'Hairdryer':22, 
'Covidcough':23, 
'Help':24, 
'Hen':25, 
'Hihat':26, 
'Hit':27, 
'Jackhammer':28, 
'Keyboardtyping':29, 
'Kissing':30, 
'Laughing':31, 
'Lighter':32, 
'Healthycough':33, 
'Manspeaking':34, 
'Metal-on-metal':35, 
'Astmacough':36, 
'Mouseclick':37, 
'Ringtone':38, 
'Rooster':39, 
'Silence':40, 
'Sitar':41, 
'Sneezing':42, 
'Snooring':43, 
'Stapler':44, 
'ToiletFlush':45, 
'Toothbrush':46, 
'Trampler':47, 
'Vaccumcleaner':48, 
'Vandalism':49, 
'WalkFootsteps':50, 
'Washingmachine':51, 
'Water':52, 
'Whimper':53, 
'Window':54, 
'HandSaw':55, 
'Siren':56, 
'Whistling':57, 
'Wind':58,'Doorknock':59
}
class CFG:
  AAL_CODE_English  = ['AlaramClock', 'Blending', 'Breaking','Canopening','Cat', 'Chirpingbirds', 'Clapping', 'Clarinet', 'Clocktick', 'Crying', 'Cupboard', 'Displaying_furniture', 'Dog', 'DoorBell','Dragonground','Drill','Drinking', 'Drum', 'Femalespeaking', 'Flute', 'Glass', 'Guitar', 'Hairdryer', 'Covidcough', 'Help', 'Hen', 'Hihat', 'Hit', 'Jackhammer', 'Keyboardtyping', 'Kissing','Laughing', 'Lighter', 'Healthycough', 'Manspeaking', 'Metal-on-metal', 'Astmacough', 'Mouseclick', 'Ringtone', 'Rooster', 'Silence', 'Sitar', 'Sneezing', 'Snooring', 'Stapler', 'ToiletFlush','Toothbrush','Trampler', 'Vaccumcleaner', 'Vandalism', 'WalkFootsteps', 'Washingmachine', 'Water', 'Whimper', 'Window', 'HandSaw', 'Siren', 'Whistling','Wind']
  AAL_CODE_german1  = ['Alarmsignal', 'Blending', 'Zerbrechen', 'Doseöffnen', 'Katze', 'ZwitscherndeVögel', 'klatschen', 'Klarinette', 'Uhr-ticken', 'Weinen', 'Schrank', 'Möbelrücken', 'Hund', 'Türklingel','Etwas-am-Boden-ziehen', 'Bohren', 'Trinken', 'Schlagzeug', 'SprechendeFrau', 'Flöte', 'Glas', 'Gitarre', 'Haartrockner', 'CovidHusten', 'Hilfe', 'Huhn', 'Schlagzeug', 'Schlag','Presslufthammer', 'Tastatur-tippen', 'Küssen', 'Lachen', 'Feuerzeug', 'GesunderHusten', 'SprechenderMann', 'Metall-auf-Metall', 'AstmaHusten', 'Mausklick', 'Klingelton', 'Hahn', 'Ruhe', 'Sitar','Niesen', 'Schnarchen', 'Tacker', 'Toilettenspülung', 'Zahnbürste', 'Trampler', 'Staubsauger', 'Vandalismus', 'Fußstapfen-gehen', 'Waschmaschine', 'Wasser', 'Wimmern', 'Fenster', 'Handsäge','Sirene', 'Pfeifen', 'Wind'] 

  


AAL_CODE_german  = {
'Alarmsignal':0, 
'Blending':1, 
'Zerbrechen':2, 
'Doseöffnen':3, 
'Katze':4, 
'ZwitscherndeVögel':5, 
'klatschen':6, 
'Klarinette':7, 
'Uhr-ticken':8, 
'Weinen':9, 
'Electronic_Zahnbürste':10, 
'Möbelrücken':11, 
'Hund':12, 
'Türklingel':13, 
'Etwas-am-Boden-ziehen':14, 
'Bohren':15, 
'Trinken':16, 
'Schlagzeug':17, 
'SprechendeFrau':18, 
'Flöte':19, 
'Glas':20, 
'Gitarre':21, 
'Haartrockner':22, 
'CovidHusten':23, 
'Hilfe':24, 
'Huhn':25, 
'Schlagzeug':26, 
'Schlag':27, 
'Presslufthammer':28, 
'Tastatur-tippen':29, 
'Küssen':30, 
'Lachen':31, 
'Feuerzeug':32, 
'GesunderHusten':33, 
'SprechenderMann':34, 
'Metall-auf-Metall':35, 
'AstmaHusten':36, 
'Mausklick':37, 
'Klingelton':38, 
'Hahn':39, 
'Ruhe':40, 
'Sitar':41, 
'Niesen':42, 
'Schnarchen':43, 
'Tacker':44, 
'Toilettenspülung':45, 
'Zahnbürste':46, 
'Trampler':47, 
'Staubsauger':48, 
'Vandalismus':49, 
'Fußstapfen-gehen':50, 
'Waschmaschine':51, 
'Wasser':52, 
'Wimmern':53, 
'Fenster':54, 
'Handsäge':55, 
'Sirene':56, 
'Pfeifen':57, 
'Wind':58,'Türklopfen':59
}

AAL_CODE_dict = {v: k for k, v in AAL_CODE.items()}
AAL_CODE_dict_german = {v: k for k, v in AAL_CODE_german.items()}
###################################################################################################################################################################################
#Feature Extraction
import torch.nn as nn
import numpy as np
import torch
import librosa
import torch.nn.functional as F
class DFTBase(nn.Module):
    def __init__(self):
        """Base class for DFT and IDFT matrix"""
        super(DFTBase, self).__init__()

    def dft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(-2 * np.pi * 1j / n)
        W = np.power(omega, x * y)
        return W

    def idft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(2 * np.pi * 1j / n)
        W = np.power(omega, x * y)
        return W
    
    
class STFT(DFTBase):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, 
        window='hann', center=True, pad_mode='reflect', freeze_parameters=True):
        """Implementation of STFT with Conv1d. The function has the same output 
        of librosa.core.stft
        """
        super(STFT, self).__init__()

        assert pad_mode in ['constant', 'reflect']

        self.n_fft = n_fft
        self.center = center
        self.pad_mode = pad_mode

        # By default, use the entire frame
        if win_length is None:
            win_length = n_fft

        # Set the default hop, if it's not already specified
        if hop_length is None:
            hop_length = int(win_length // 4)

        fft_window = librosa.filters.get_window(window, win_length, fftbins=True)

        # Pad the window out to n_fft size
        fft_window = librosa.util.pad_center(fft_window, size=n_fft)

        # DFT & IDFT matrix
        self.W = self.dft_matrix(n_fft)

        out_channels = n_fft // 2 + 1

        self.conv_real = nn.Conv1d(in_channels=1, out_channels=out_channels, 
            kernel_size=n_fft, stride=hop_length, padding=0, dilation=1, 
            groups=1, bias=False)

        self.conv_imag = nn.Conv1d(in_channels=1, out_channels=out_channels, 
            kernel_size=n_fft, stride=hop_length, padding=0, dilation=1, 
            groups=1, bias=False)

        self.conv_real.weight.data = torch.Tensor(
            np.real(self.W[:, 0 : out_channels] * fft_window[:, None]).T)[:, None, :]
        # (n_fft // 2 + 1, 1, n_fft)

        self.conv_imag.weight.data = torch.Tensor(
            np.imag(self.W[:, 0 : out_channels] * fft_window[:, None]).T)[:, None, :]
        # (n_fft // 2 + 1, 1, n_fft)

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        """input: (batch_size, data_length)
        Returns:
          real: (batch_size, n_fft // 2 + 1, time_steps)
          imag: (batch_size, n_fft // 2 + 1, time_steps)
        """

        x = input[:, None, :]   # (batch_size, channels_num, data_length)

        if self.center:
            x = F.pad(x, pad=(self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)

        real = self.conv_real(x)
        imag = self.conv_imag(x)
        # (batch_size, n_fft // 2 + 1, time_steps)

        real = real[:, None, :, :].transpose(2, 3)
        imag = imag[:, None, :, :].transpose(2, 3)
        # (batch_size, 1, time_steps, n_fft // 2 + 1)

        return real, imag
    
    
class Spectrogram(nn.Module):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, 
        window='hann', center=True, pad_mode='reflect', power=2.0, 
        freeze_parameters=True):
        """Calculate spectrogram using pytorch. The STFT is implemented with 
        Conv1d. The function has the same output of librosa.core.stft
        """
        super(Spectrogram, self).__init__()

        self.power = power

        self.stft = STFT(n_fft=n_fft, hop_length=hop_length, 
            win_length=win_length, window=window, center=center, 
            pad_mode=pad_mode, freeze_parameters=True)

    def forward(self, input):
        """input: (batch_size, 1, time_steps, n_fft // 2 + 1)
        Returns:
          spectrogram: (batch_size, 1, time_steps, n_fft // 2 + 1)
        """

        (real, imag) = self.stft.forward(input)
        # (batch_size, n_fft // 2 + 1, time_steps)

        spectrogram = real ** 2 + imag ** 2

        if self.power == 2.0:
            pass
        else:
            spectrogram = spectrogram ** (power / 2.0)

        return spectrogram

    
class LogmelFilterBank(nn.Module):
    def __init__(self, sr=32000, n_fft=2048, n_mels=64, fmin=50, fmax=14000, is_log=True, 
        ref=1.0, amin=1e-10, top_db=80.0, freeze_parameters=True):
        """Calculate logmel spectrogram using pytorch. The mel filter bank is 
        the pytorch implementation of as librosa.filters.mel 
        """
        super(LogmelFilterBank, self).__init__()

        self.is_log = is_log
        self.ref = ref
        self.amin = amin
        self.top_db = top_db

        self.melW = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
            fmin=fmin, fmax=fmax).T
        # (n_fft // 2 + 1, mel_bins)

        self.melW = nn.Parameter(torch.Tensor(self.melW))

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        """input: (batch_size, channels, time_steps)
        
        Output: (batch_size, time_steps, mel_bins)
        """

        # Mel spectrogram
        mel_spectrogram = torch.matmul(input, self.melW)

        # Logmel spectrogram
        if self.is_log:
            output = self.power_to_db(mel_spectrogram)
        else:
            output = mel_spectrogram

        return output


    def power_to_db(self, input):
        """Power to db, this function is the pytorch implementation of 
        librosa.core.power_to_lb
        """
        ref_value = self.ref
        log_spec = 10.0 * torch.log10(torch.clamp(input, min=self.amin, max=np.inf))
        log_spec -= 10.0 * np.log10(np.maximum(self.amin, ref_value))

        if self.top_db is not None:
            if self.top_db < 0:
                raise ParameterError('top_db must be non-negative')
            log_spec = torch.clamp(log_spec, min=log_spec.max().item() - self.top_db, max=np.inf)

        return log_spec


class DropStripes(nn.Module):
    def __init__(self, dim, drop_width, stripes_num):
        """Drop stripes. 
        Args:
          dim: int, dimension along which to drop
          drop_width: int, maximum width of stripes to drop
          stripes_num: int, how many stripes to drop
        """
        super(DropStripes, self).__init__()

        assert dim in [2, 3]    # dim 2: time; dim 3: frequency

        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num

    def forward(self, input):
        """input: (batch_size, channels, time_steps, freq_bins)"""

        assert input.ndimension() == 4

        if self.training is False:
            return input

        else:
            batch_size = input.shape[0]
            total_width = input.shape[self.dim]

            for n in range(batch_size):
                self.transform_slice(input[n], total_width)

            return input


    def transform_slice(self, e, total_width):
        """e: (channels, time_steps, freq_bins)"""

        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.drop_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]

            if self.dim == 2:
                e[:, bgn : bgn + distance, :] = 0
            elif self.dim == 3:
                e[:, :, bgn : bgn + distance] = 0


class SpecAugmentation(nn.Module):
    def __init__(self, time_drop_width, time_stripes_num, freq_drop_width, 
        freq_stripes_num):
        """Spec augmetation. 
        [ref] Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D. 
        and Le, Q.V., 2019. Specaugment: A simple data augmentation method 
        for automatic speech recognition. arXiv preprint arXiv:1904.08779.
        Args:
          time_drop_width: int
          time_stripes_num: int
          freq_drop_width: int
          freq_stripes_num: int
        """

        super(SpecAugmentation, self).__init__()

        self.time_dropper = DropStripes(dim=2, drop_width=time_drop_width, 
            stripes_num=time_stripes_num)

        self.freq_dropper = DropStripes(dim=3, drop_width=freq_drop_width, 
            stripes_num=freq_stripes_num)

    def forward(self, input):
        x = self.time_dropper(input)
        x = self.freq_dropper(x)
        return x
###################################################################################################################################################################################
'''

From SED Paper Qui


'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.

    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1:, :].repeat(
        1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class AttBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear",
                 temperature=1.0):
        super().__init__()

        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.bn_att = nn.BatchNorm1d(out_features)
        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)

    def forward(self, x):
        # x: (n_samples, n_in, n_time){self.fold}
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
        
class PANNsAALAtt(nn.Module):
    def __init__(self, sample_rate: int, window_size: int, hop_size: int,
                 mel_bins: int, fmin: int, fmax: int, classes_num: int, apply_aug: bool, top_db=None):
        super().__init__()
        
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        self.interpolate_ratio = 32  # Downsampled ratio
        self.apply_aug = apply_aug

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.att_block = AttBlock(1024, classes_num, activation='sigmoid')


        self.densenet_features = models.densenet121(pretrained=False).features

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        
    def cnn_feature_extractor(self, x):
        x = self.densenet_features(x)
        return x
    
    def preprocess(self, input_x, mixup_lambda=None):

        x = self.spectrogram_extractor(input_x)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.apply_aug:
            x = self.spec_augmenter(x)

        return x, frames_num
        

    def forward(self, input_data):
        input_x, mixup_lambda = input_data
        """
        Input: (batch_size, data_length)"""
        b, c, s = input_x.shape
        input_x = input_x.reshape(b*c, s)
        x, frames_num = self.preprocess(input_x, mixup_lambda=mixup_lambda)
        if mixup_lambda is not None:
            b = (b*c)//2
            c = 1
        # Output shape (batch size, channels, time, frequency)
        x = x.expand(x.shape[0], 3, x.shape[2], x.shape[3])
        x = self.cnn_feature_extractor(x)
        
        # Aggregate in frequency axis
        x = torch.mean(x, dim=3)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output,
                                       self.interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)
        frame_shape =  framewise_output.shape
        clip_shape = clipwise_output.shape
        output_dict = {
            
            'clipwise_output': clipwise_output.reshape(b, c, clip_shape[1]),
            'framewise_output': framewise_output.reshape(b, c, frame_shape[1],frame_shape[2]),
        }

        return output_dict
    
    
#Get model
def get_model(ModelClass: object, config: dict, weights_path: str):
    model = ModelClass(**config)
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model

#Model settings
list_of_models={
        "model_class": PANNsAALAtt,
        "config": {
            "sample_rate": 44100,
            "window_size": 1024,
            "hop_size": 320,
            "mel_bins": 64,
            "fmin": 50,
            "fmax": 32000,
            "classes_num": 60,
            "apply_aug": True,
            "top_db": None
        },
        "weights_path": "new_models_60classes_mixup_12thdecember_00_6AM/final_5fold_sed_dense121_nomix_fold0/final_5fold_sed_dense121_nomix_fold0_checkpoint_84_score=0.9421.pt",
        "clip_threshold": 0.5,
        "threshold": 0.5
}
    

PERIOD = 4
SR = 44100

TTA = 1
list_of_models['model'] = get_model(list_of_models["model_class"], list_of_models["config"], list_of_models["weights_path"])
#####################################################################################################################################################################################

def dt_local():
    
    now = time.time()
       
    st = datetime.datetime.fromtimestamp(now).strftime("%Y-%m-%d %I:%M:%S %p")
    return st
#print (dt_local())  



def dt_local_2():
    
    now = time.time()
       
    st = datetime.datetime.fromtimestamp(now).strftime("%Y-%m-%d %I:%M:00 %p")
    return st
#print (dt_local())     
######################################################################################################################################################################################
TEST_FUNCTIONS = []
LAST_DURATION = 0

PDATA = []
SCOUNT = {}

FRAMES = np.array([], dtype='float32')


def openStream():      
        # Setup pyaudio
        paudio = pyaudio.PyAudio()
       
        # Stream Settings
        stream = paudio.open(format=pyaudio.paFloat32,
                            #input_device_index=0,Sampathkumar
                            channels=1,
                            rate=SR,
                            input=True,
                            frames_per_buffer=SR // 2)

        return stream


def record():

    global FRAMES

    # Open stream
    stream = openStream()

    while True:

        try:

            # Read from stream
            data = stream.read(SR // 2)
            data = np.fromstring(data, 'float32');
            FRAMES = np.concatenate((FRAMES, data))

            # Truncate frame countFRAMES[-int(cfg['SAMPLE_RATE'] * cfg['SPEC_LENGTH']):]
            FRAMES = FRAMES[-int(SR * PERIOD):]
        except KeyboardInterrupt:
            KILL_ALL= True
            break
        except:
            FRAMES = np.array([], dtype='float32')
            stream = openStream()
            continue

def analyzeStream(model,threshold,clip_threshold):

    global scnt
    global mqtt_time_last_send
    global mqtt_debounce

    # Time
    start = time.time()
    
    clip = FRAMES.copy()
    if len(clip) < SR*4:
       clip=clip 

    
    p_final={}
    p_finals={}
    spec_batch = []
    p_final={}
    audios = []
    y = clip.astype(np.float32)
    len_y = len(y)
    start = 0
    end = PERIOD * SR
    while True:
        
        y_batch = y[start:end].astype(np.float32)
        if len(y_batch) != PERIOD * SR:
            y_pad = np.zeros(PERIOD * SR, dtype=np.float32)
            y_pad[:len(y_batch)] = y_batch
            audios.append(y_pad)
            break
        start = end
        end += PERIOD * SR
        audios.append(y_batch)
        
        

        
    array = np.asarray(audios)
    tensors = torch.from_numpy(array)
    
    model.eval()
    global_time = 0.0
    
    for image in tensors:

        image = image.unsqueeze(0).unsqueeze(0)
        image = image.expand(image.shape[0], TTA, image.shape[2])
        image = image.to(device)
        
        with torch.no_grad():
           prediction = model((image, None))

           clipwise_outputs = prediction["clipwise_output"].detach(
                ).cpu().numpy()[0].mean(axis=0)
                
        now = time.time()
        if now > mqtt_time_last_send + mqtt_msg_freq and len(mqtt_debounce) > 0:
            e_z = np.exp(np.average(np.log(mqtt_debounce), axis=0))
            res = e_z / sum(e_z)
            mqtt_message = '{{"timestamp":{},"sampleRate":{},"prediction":{{{}}}}}'.format(now * 1000, mqtt_msg_freq * 1000, ','.join([f'"{AAL_CODE_dict_german[40] if idx == 17 else AAL_CODE_dict_german[idx]}":{output}' for idx, output in enumerate(res)]))
            # print("[MQTT] sending \"%s\" on \"%s\"" % (mqtt_topic, mqtt_message))
            mqtt_client.publish(mqtt_topic, mqtt_message)
            mqtt_time_last_send = now
            mqtt_debounce.clear()
        else:
            mqtt_debounce.append(clipwise_outputs)
        
        clip_thresholded = clipwise_outputs >= clip_threshold
        clip_indices = np.argwhere(clip_thresholded).reshape(-1)
       
        st=dt_local()
        st2=dt_local_2()

        for ci in clip_indices:
            #clip_codes.append(AAL_CODE_dict[ci])
            if ci==17:
               ci=40
            else:
               ci
               
            #if AAL_CODE_dict[ci]=='Dog':
            #       pred = {'ClassName':'Silence','ClassName_German':'Ruhe','Datetime':st,'Confidence':float(np.max(clipwise_outputs))}
            #elif AAL_CODE_dict[ci]=='Jackhammer':
            #       pred = {'ClassName':'Silence','ClassName_German':'Ruhe','Datetime':st,'Confidence':float(np.max(clipwise_outputs))}
            #elif AAL_CODE_dict[ci]=='Water':
            #       pred = {'ClassName':'Drinking','ClassName_German':'Trinken','Datetime':st,'Confidence':float(np.max(clipwise_outputs))}
            #elif AAL_CODE_dict[ci]=='Hi-hat':
            #       pred = {'ClassName':'WalkFootsteps','ClassName_German':'Fußstapfen-gehen','Datetime':st,'Confidence':float(np.max(clipwise_outputs))}
            #elif AAL_CODE_dict[ci]=='Covidcough':
            #       pred = {'ClassName':'Healthycough','ClassName_German':'GesunderHusten','Datetime':st,'Confidence':float(np.max(clipwise_outputs))}
            #elif AAL_CODE_dict[ci]=='Astmacough':
            #      pred = {'ClassName':'Healthycough','ClassName_German':'GesunderHusten','Datetime':st,'Confidence':float(np.max(clipwise_outputs))}
            #elif AAL_CODE_dict[ci]=='Cupboard':
            #       pred = {'ClassName':'Silence','ClassName_German':'Ruhe','Datetime':st,'Confidence':float(np.max(clipwise_outputs))}    
                     
            #
            
            #elif AAL_CODE_dict[ci]=='Siren':
            #       pred = {'ClassName':'Whistling','ClassName_German':'Pfeifen','Datetime':st,'Confidence':float(np.max(clipwise_outputs))}
            #elif AAL_CODE_dict[ci]=='Kissing':
            #       pred = {'ClassName':'Drinking','ClassName_German':'Trinken','Datetime':st,'Confidence':float(np.max(clipwise_outputs))}

           
            #else:
            pred = {'ClassName':AAL_CODE_dict[ci],'ClassName_German':AAL_CODE_dict_german[ci],'Datetime':st,'Datetime_2':st2,'Confidence':float(np.max(clipwise_outputs))}
            # print(pred)
            # mycol.insert(pred)


            label=AAL_CODE_dict_german[ci]
            #if label=='Hund':
            #       label = 'Ruhe'
            #elif label=='Presslufthammer':
            #       label ='Ruhe'
            #elif label=='Wasser':
            #       label ='Trinken'
            #elif label=='Schlagzeug':
            #       label ='Fußstapfen-gehen'
            #elif label=='CovidHusten':
            #       label ='GesunderHusten'
            #elif label=='AstmaHusten':
            #       label ='GesunderHusten'
            #elif label=='Sirene':
            #       label ='Pfeifen'

            #elif label=='Küssen':
            #       label ='Trinken'
            #elif label=='Möbelrücken':
            #       label ='Trinken'
       
        #print (label)
            conf=np.max(clipwise_outputs)
        #print(conf)
        #if conf<=0.2:
        #     conf="0.90"
        #     label="Silence"
             
        #else:
        #     conf=np.max(clipwise_outputs)
        #     label=AAL_CODE_dict[np.argmax(clipwise_outputs)]

            p_labels = {}
            p_labels[label] = AAL_CODE_dict_german[ci]
            p_sorted =  sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)
    #print (p_sorted)
        #print(Samples)
        #mycol.insert(Samples)
            for i in range(len(p_sorted)):
             label = p_sorted[i][0]

        
             p_final[i] = {'species':label, 'score':str(float(np.max(clipwise_outputs))),'Datetime':st}
             #p_finals[i] = {'labels':label, 'score':str(np.max(clipwise_outputs)),'Datetime':st}
              #print(p_f)


 
   
    data = {'prediction': {}}
    data['prediction']['0'] = p_final
    
    #print (data)
    data['time'] = time.time() - start
    
    return data



def resultPooling(data):

    global PDATA

    # Load counter data
    if len(SCOUNT) == 0:
        loadCounter()

    # Get prediction from data
    prediction = data['prediction']['0']

    # Add to global data
    PDATA.append(prediction)

    # Trim to max size
    PDATA = PDATA[-cfg.RESULT_POOLING:]

    # Collect scores from every prediction
    scores = {}
    for p in PDATA:
        for rank in p:
            species = p[rank]['species']
            score = (float(p[rank]['score']) * cfg.SCORE_MULTIPLY)**2
            if species in scores:
                scores[species].append(score)
            else:
                scores[species] = [score]

    # Calculate average for every class
    for species in scores:
        if not species in cfg.BLACK_LIST:
            scores[species] = min(1.0, np.mean(np.array(scores[species]) ** 1))
        else:
            scores[species] = 0.0

    # Remove counts older than one hour
    for species in SCOUNT:
        if len(SCOUNT[species]) > 0 and time.time() - float(SCOUNT[species][0]) > 3600:
            del SCOUNT[species][0]

    # Prepare counter dict
    scount_temp = copy.deepcopy(SCOUNT)
    for species in scount_temp:
        scount_temp[species] = len(scount_temp[species])

    # Count species occurences    
    for species in scores:
        if scores[species] > cfg.COUNT_THRESHOLD:

            # Do we have an entry?
            if species in SCOUNT and len(SCOUNT[species]) > 0:                

                # Only count every few seconds
                if time.time() - float(SCOUNT[species][-1]) > cfg.COUNT_TIMEOUT:
                    SCOUNT[species].append(str(time.time()))
                    #saveSpeciesCountData(species, time.time())

                # Temp counter
                scount_temp[species] = len(SCOUNT[species])

                # Status
                #log.i(('COUNTING:', species, len(SCOUNT[species])), discard=True)
                    
            else:
                SCOUNT[species] = [str(time.time())]
                scount_temp[species] = 1
                #saveSpeciesCountData(species, time.time())

    # Sort species count
    scount_sorted = sorted(scount_temp.items(), key=operator.itemgetter(1), reverse=True)

    # Save counter data
    # saveCounter()

    # Re-rank based on avg score
    s_sorted =  sorted(scores.items(), key=operator.itemgetter(1), reverse=True)[:cfg.MAX_RESULTS]

    # Convert to dict
    s_dict = {}
    for i in range(len(s_sorted)):
        s_dict[i] = {'species':s_sorted[i][0], 'score':str(s_sorted[i][1])}
        #print (s_dict[i])

    # free memory
    del scount_temp

    # Return average data
    data['prediction']['0'] = s_dict
    data['counts'] = scount_sorted
    return data

####################### HELPER ########################

def loadCounter():

    global SCOUNT

    try:
        with open(cfg.COUNTER_FILE, 'r') as cfile:
            SCOUNT = json.load(cfile)

    except:
        SCOUNT = {}

def saveCounter():

    with open(cfg.COUNTER_FILE, 'w') as cfile:
        json.dump(SCOUNT, cfile)


    

def showResults(data, max_results=8, min_score=0.01):

    # Timestamp
    #print (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), 'TIME FOR PREDICTION:', int(LAST_DURATION * 1000), 'PREDICTIONS:')

    # Parse predictions
    for entry in data['prediction']['0']:
        d = data['prediction']['0'][entry]
        if float(d['score']) >= min_score:
            s = '\t' + str(int(float(d['score']) * 100)).zfill(3) + '% - ' + d['species'].split(';')[0] + '\n'
            #log.r(s, discard=True)



#########################  MAIN  ###########################
def run():

    # Start recording
    log.i(('STARTING RECORDING WORKER'))
    recordWorker = Thread(target=record, args=())
    recordWorker.start()

    # Keep running...
    log.i(('STARTING ANALYSIS'))
    while not cfg.KILL_ALL:

        try:

            # Make prediction
            
            p = analyzeStream(model=list_of_models['model'],threshold=list_of_models["threshold"],clip_threshold=list_of_models["clip_threshold"])
            data = resultPooling(p)

            # Write analysis to file
            # with open('clo_analysis.json', 'w') as afile:
            #     json.dump(data, afile)

            # DEBUG: Show prediction
            showResults(data)
                        

            # Sleep if we are too fast
            if 'time_for_analysis' in p and p['time_for_analysis'] < 0.5:
                #time.sleep(0.5 - p['time_for_analysis'])
                 time.sleep(5)  
        except KeyboardInterrupt:
            cfg.KILL_ALL = True
            break
        #except:
            #continue
            #cfg.KILL_ALL = True

    # Done
    log.i(('TERMINATED'))
    
if __name__ == '__main__':

    #load()
    #buildModel()
    #model=load_model()
    run()
