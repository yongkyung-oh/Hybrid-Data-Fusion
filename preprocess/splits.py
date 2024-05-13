import os
import sys
import time
import math
import copy
import pickle
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchsummary

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms

from tqdm import tqdm
from datetime import datetime, timedelta

import warnings
from pandas.core.common import SettingWithCopyWarning

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, mean_absolute_error

from sktime.datasets import load_UCR_UEA_dataset
from sktime.utils.data_processing import from_2d_array_to_nested
from sktime_dl.regression import *

# Set random seed
SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:1" if use_cuda else "cpu")
device = torch.device("cuda" if use_cuda else "cpu")

# check path
if not os.path.exists('splits'):
    os.mkdir('splits')

# get data path
data_path = "/media/usr/HDD/Data/Lifelog/"

df_user_info = pd.read_csv(os.path.join(data_path, 'user_info_2020.csv'), index_col=0)
df_user_sleep = pd.read_csv(os.path.join(data_path, 'user_sleep_2020.csv'), index_col=0)
df_user_survey = pd.read_csv(os.path.join(data_path, 'user_survey_2020.csv'), index_col=0)

# get selected user info
path_dict ={
    'user01': 'user01-06', 'user02': 'user01-06', 'user03': 'user01-06', 'user04': 'user01-06', 'user05': 'user01-06', 'user06': 'user01-06', 
    'user07': 'user07-10', 'user08': 'user07-10', 'user09': 'user07-10', 'user10': 'user07-10',
    'user11': 'user11-12', 'user12': 'user11-12', 
    'user21': 'user21-25', 'user22': 'user21-25', 'user23': 'user21-25', 'user24': 'user21-25', 'user25': 'user21-25', 
    'user26': 'user26-30', 'user27': 'user26-30', 'user28': 'user26-30', 'user29': 'user26-30', 'user30': 'user26-30', 
}

# action label dict
label_dict = {
    'sleep': 0, 
    'personal_care': 1, 
    'work': 2, 
    'study': 3, 
    'household': 4, 
    'care_housemem': 5, 
    'recreation_media': 6, 
    'entertainment': 7, 
    'outdoor_act': 8, 
    'hobby': 9, 
    'recreation_etc': 10, 
    'shop': 11, 
    'community_interaction': 12, 
    'travel': 13, 
    'meal': 14, 
    'socialising': 15
}

# user_list = list(path_dict.keys())
user_list = sorted(os.listdir('sktime'))

for user_id in tqdm(user_list):
    if not os.path.exists('splits/{}'.format(user_id)):
        os.mkdir('splits/{}'.format(user_id))
    
    with open('sktime/{}'.format(user_id), 'rb') as f:
        data = pickle.load(f)

    X_data = data[0]
    y_data = data[1]

    X_data = X_data.reset_index(drop=True)
    y_data = y_data[:,0]
    
    for k in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.8, stratify=y_data)

        train_idx = X_train.index
        test_idx = X_test.index

        with open('splits/{}/{}'.format(user_id, k), 'wb') as f:
            pickle.dump([train_idx, test_idx], f)