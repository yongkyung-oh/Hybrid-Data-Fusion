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
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from sktime.datasets import load_UCR_UEA_dataset
from sktime.utils.data_processing import from_2d_array_to_nested
from sktime_dl.classification import *

# Set random seed
SEED = 0
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
if not os.path.exists('out_clf'):
    os.mkdir('out_clf')

# get data path
data_path = "/media/usr/HDD/Data/Lifelog/"

df_user_info = pd.read_csv(os.path.join(data_path, 'user_info_2020.csv'), index_col=0)
df_user_sleep = pd.read_csv(os.path.join(data_path, 'user_sleep_2020.csv'), index_col=0)
df_user_survey = pd.read_csv(os.path.join(data_path, 'user_survey_2020.csv'), index_col=0)

# get selected user info
path_dict = {
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

## define modules
# define dataset
class MM_Dataset(Dataset):
    def __init__(self, tab_all, pred_all, true_all):
        self.tab_all = tab_all
        self.pred_all = pred_all
        self.true_all = true_all
        
    def __len__(self):
        return len(self.tab_all)
    
    def __getitem__(self, idx):
        tab_info = self.tab_all[idx]
        tab_features = torch.FloatTensor(tab_info.astype(float))
        
        pred_all = self.pred_all[idx]
        true_all = self.true_all[idx]
                
        sample = {'target': true_all, 'pred': pred_all, 'feature': tab_features}
        return sample

class PredictionModel(nn.Module):
    def __init__(self, seq_features=22, tab_features=22, emb_dim=32, hidden=128, num_class=10):
        super().__init__()

        self.feature_net = nn.Linear(tab_features,tab_features)
        self.emb = nn.Linear(tab_features,emb_dim)
        self.mlp = nn.Sequential(nn.Linear(emb_dim,hidden),
                                 nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(hidden,num_class))
        
    def forward(self, feature, pred):
        feature = self.feature_net(feature)
        x = self.emb(feature)
        out = self.mlp(x)
        return out

def train(model, optimizer, criterion, train_iter):
    model.train()
    total_loss = 0
    for batch in train_iter:
        y = batch['target'].long().to(device)
        optimizer.zero_grad()

        logit = model(batch['feature'].to(device), batch['pred'].to(device))
        loss = criterion(logit, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
    size = len(train_iter.dataset)
    avg_loss = total_loss / size
    return avg_loss

def evaluate(model, criterion, val_iter):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_iter:
            y = batch['target'].long().to(device)

            logit = model(batch['feature'].to(device), batch['pred'].to(device))
            loss = criterion(logit, y)
            
            total_loss += loss.item() * len(y)
    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    return avg_loss


# user-based modeling
# user_list = list(path_dict.keys())
user_list = sorted(os.listdir('/media/usr/WORKING/HAR/Lifelog_HAR/sktime'))

for user_id in user_list[::-1]:
    if not os.path.exists('out_clf/{}'.format(user_id)):
        os.mkdir('out_clf/{}'.format(user_id))

    with open('/media/usr/WORKING/HAR/Lifelog_HAR/preprocess/{}'.format(user_id), 'rb') as f:
        data_raw = pickle.load(f)
        
    with open('/media/usr/WORKING/HAR/Lifelog_HAR/sktime/{}'.format(user_id), 'rb') as f:
        data = pickle.load(f)

    X_data = data[0]
    y_data = data[1]

    X_data = X_data.reset_index(drop=True)
    y_data = y_data[:,0] 
    
    for k in range(10):
        # X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.8, stratify=y_data)
                
        with open('/media/usr/WORKING/HAR/Lifelog_HAR/splits/{}/{}'.format(user_id, k), 'rb') as f:
            train_idx, test_idx = pickle.load(f)

        X_train = X_data.loc[train_idx]
        X_test = X_data.loc[test_idx]

        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        
        y_train = y_data[train_idx]
        y_test = y_data[test_idx]
            
        Z_train = np.array(data_raw, dtype=object)[train_idx]
        Z_test = np.array(data_raw, dtype=object)[test_idx]
        
        for modelname in ['CNN', 'FCN', 'Inception', 'LSTMFCN', 'Encoder', 'MACNN', 'MCDCNN', 'MLP', 'ResNet', 'TapNet', 'TLENET']:
            batch_size = 64
            nb_epoch = 10
            if modelname == 'CNN': 
                network = CNNClassifier(nb_epochs=nb_epoch, batch_size=batch_size, verbose=False)
            elif modelname == 'FCN':
                network = FCNClassifier(nb_epochs=nb_epoch, batch_size=batch_size, verbose=False)
            elif modelname == 'Inception':
                network = InceptionTimeClassifier(nb_epochs=nb_epoch, batch_size=batch_size, verbose=False)
            elif modelname == 'LSTMFCN':
                network = LSTMFCNClassifier(nb_epochs=nb_epoch, batch_size=batch_size, verbose=False)
            elif modelname == 'Encoder':
                network = EncoderClassifier(nb_epochs=nb_epoch, batch_size=batch_size, verbose=False)
            elif modelname == 'MACNN':
                network = MACNNClassifier(nb_epochs=nb_epoch, batch_size=batch_size, verbose=False)
            elif modelname == 'MCDCNN':
                network = MCDCNNClassifier(nb_epochs=nb_epoch, batch_size=batch_size, verbose=False)
            elif modelname == 'MLP':
                network = MLPClassifier(nb_epochs=nb_epoch, batch_size=batch_size, verbose=False)
            elif modelname == 'ResNet':
                network = ResNetClassifier(nb_epochs=nb_epoch, batch_size=batch_size, verbose=False)
            elif modelname == 'TapNet':
                network = TapNetClassifier(nb_epochs=nb_epoch, batch_size=batch_size, verbose=False)
            elif modelname == 'TLENET':
                network = TLENETClassifier(nb_epochs=nb_epoch, batch_size=batch_size, verbose=False)

            network.fit(X_train, y_train)
            y_pred = network.predict(X_test)
            y_pred_proba = network.predict_proba(X_test)

            # with open(os.path.join('out_clf', user_id, '{}_{}'.format(modelname, k)), 'wb') as f:
            #     pickle.dump([y_test, y_pred, y_pred_proba], f)

            print(user_id, modelname, accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'))
            
            
            ## Train MM model
            # get tabular features
            tab_train = np.hstack([np.vstack(Z_train[:,1]), np.vstack(Z_train[:,2])])
            tab_test = np.hstack([np.vstack(Z_test[:,1]), np.vstack(Z_test[:,2])])

            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(tab_train)

            tab_train = scaler.transform(tab_train)
            tab_test = scaler.transform(tab_test)

            # get sequence based prediction
            y_pred_train = network.predict_proba(X_train)
            y_pred_test = network.predict_proba(X_test)

            # define dataset
            y_emb = dict(zip(np.unique(y_train), range(len(np.unique(y_train)))))
            y_rev = dict(zip(range(len(np.unique(y_train))), np.unique(y_train)))

            y_train_emb = np.array([y_emb[y] for y in y_train])
            y_test_emb = np.array([y_emb[y] for y in y_test])
            
            train_dataset = MM_Dataset(tab_train, y_pred_train, y_train_emb)
            test_dataset = MM_Dataset(tab_test, y_pred_test, y_test_emb)

            train_batch = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)
            test_batch = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)
            
            # train model
            model = PredictionModel(num_class=y_pred_train.shape[1])
            model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            criterion = nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

            best_loss = 100
            for e in tqdm(range(1, 100 + 1)):
                train_loss = train(model, optimizer, criterion, train_batch)
                test_loss = evaluate(model, criterion, test_batch)

                if e % 10 == 0:
                    print("[Epoch: %03d] train loss : %5.5f | test loss : %5.5f" % (e, train_loss, test_loss))

                if test_loss < best_loss:
                    best_loss = test_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

                scheduler.step()

            # prediction    
            model.load_state_dict(best_model_wts)
            model.eval()

            total_loss = 0
            y_true_mm, y_pred_mm = [], []
            y_logit_mm = []
            with torch.no_grad():
                for batch in test_batch:
                    y = batch['target'].long().to(device)

                    logit = model(batch['feature'].to(device), batch['pred'].to(device))
                    logit = logit + batch['pred'].to(device) # late fusion
                    loss = criterion(logit, y)
                    # pred = F.softmax(logit, dim=1).max(1)[1]
                    pred = logit.max(1)[1]

                    total_loss += loss.item() * len(y)

                    y_true_mm.append(y.cpu().numpy())
                    y_pred_mm.append(pred.cpu().numpy())
                    y_logit_mm.append(logit.cpu().numpy())

            size = len(test_batch.dataset)
            avg_loss = total_loss / size

            y_true_mm = np.array([x for y in y_true_mm for x in y]).flatten()
            y_pred_mm = np.array([x for y in y_pred_mm for x in y]).flatten()
            y_logit_mm = np.array([x for y in y_logit_mm for x in y]).flatten()

            y_true_mm = np.array([y_rev[y] for y in y_true_mm])
            y_pred_mm = np.array([y_rev[y] for y in y_pred_mm])
            
            with open(os.path.join('out_clf', user_id, '{}_{}'.format(modelname, k)), 'wb') as f:
                pickle.dump([y_test, y_pred, y_pred_proba, y_true_mm, y_pred_mm, y_logit_mm], f)
                
            print(user_id, modelname, accuracy_score(y_true_mm, y_pred_mm), f1_score(y_true_mm, y_pred_mm, average='weighted'))
