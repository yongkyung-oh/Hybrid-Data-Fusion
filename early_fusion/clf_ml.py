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

# import torch
# import torchvision
# import torchsummary

from tqdm import tqdm
# from torch.utils.data import Dataset, DataLoader
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, models, transforms

from tqdm import tqdm
from datetime import datetime, timedelta

import warnings
from pandas.core.common import SettingWithCopyWarning

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, mean_absolute_error

# from sktime.datasets import load_UCR_UEA_dataset
# from sktime.utils.data_processing import from_2d_array_to_nested
# from sktime_dl.regression import *

# Set random seed
SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# # CUDA for PyTorch
# use_cuda = torch.cuda.is_available()
# # device = torch.device("cuda:1" if use_cuda else "cpu")
# device = torch.device("cuda" if use_cuda else "cpu")

import tsfel
from pycaret.classification import *

# get data path
data_path = "/media/usr/HDD/Data/Lifelog/"

# check path
if not os.path.exists('out_clf_ml'):
    os.mkdir('out_clf_ml')

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

# user-based modeling
# user_list = list(path_dict.keys())
user_list = sorted(os.listdir('/media/usr/WORKING/HAR/Lifelog_HAR/sktime'))

for user_id in user_list[::-1]:
    if not os.path.exists('out_clf_ml/{}'.format(user_id)):
        os.mkdir('out_clf_ml/{}'.format(user_id))

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
        
        X_train = X_train.applymap(lambda s: np.array(s)).to_numpy()
        X_test = X_test.applymap(lambda s: np.array(s)).to_numpy()
        
        y_train = y_data[train_idx]
        y_test = y_data[test_idx]
            
        Z_train = np.array(data_raw, dtype=object)[train_idx]
        Z_test = np.array(data_raw, dtype=object)[test_idx]
        
        # seq values
        seq_train = []
        for seq in tqdm(X_train):
            seq = np.stack(seq)
            seq_features = []
            for s in seq:
                s_features = [
                    tsfel.feature_extraction.features.autocorr(s),         
                    tsfel.feature_extraction.features.zero_cross(s),         
                    tsfel.feature_extraction.features.calc_min(s),         
                    tsfel.feature_extraction.features.calc_max(s),         
                    tsfel.feature_extraction.features.calc_median(s),         
                    tsfel.feature_extraction.features.calc_mean(s),         
                    tsfel.feature_extraction.features.calc_std(s),
                    tsfel.feature_extraction.features.median_abs_deviation(s),         
                    tsfel.feature_extraction.features.max_frequency(s, fs=1),         
                    tsfel.feature_extraction.features.median_frequency(s, fs=1),         
                ]
                seq_features.append(s_features)
            seq_train.append(np.hstack(seq_features))

        # seq values
        seq_test = []
        for seq in tqdm(X_test):
            seq = np.stack(seq)
            seq_features = []
            for s in seq:
                s_features = [
                    tsfel.feature_extraction.features.autocorr(s),         
                    tsfel.feature_extraction.features.zero_cross(s),         
                    tsfel.feature_extraction.features.calc_min(s),         
                    tsfel.feature_extraction.features.calc_max(s),         
                    tsfel.feature_extraction.features.calc_median(s),         
                    tsfel.feature_extraction.features.calc_mean(s),         
                    tsfel.feature_extraction.features.calc_std(s),
                    tsfel.feature_extraction.features.median_abs_deviation(s),         
                    tsfel.feature_extraction.features.max_frequency(s, fs=1),         
                    tsfel.feature_extraction.features.median_frequency(s, fs=1),         
                ]
                seq_features.append(s_features)
            seq_test.append(np.hstack(seq_features))
            
        ## Train MM model
        # get tabular features
        tab_train = np.hstack([np.vstack(Z_train[:,1]), np.vstack(Z_train[:,2])])
        tab_test = np.hstack([np.vstack(Z_test[:,1]), np.vstack(Z_test[:,2])])

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(tab_train)

        tab_train = scaler.transform(tab_train)
        tab_test = scaler.transform(tab_test)

        # define dataset
        y_emb = dict(zip(np.unique(y_train), range(len(np.unique(y_train)))))
        y_rev = dict(zip(range(len(np.unique(y_train))), np.unique(y_train)))

        y_train_emb = np.array([y_emb[y] for y in y_train])
        y_test_emb = np.array([y_emb[y] for y in y_test])
        
        train_df = pd.DataFrame(np.hstack([y_train_emb.reshape(-1,1), seq_train, tab_train]))
        test_df = pd.DataFrame(np.hstack([y_test_emb.reshape(-1,1), seq_test, tab_test]))

        train_df = train_df.rename({0:'target'}, axis=1)
        test_df = test_df.rename({0:'target'}, axis=1)

        # Setup
        exp_clf = setup(data = train_df, target = 'target', session_id=SEED, test_data=test_df,
                        train_size = 0.8, data_split_stratify = False, fold = 5, n_jobs=32,
                        preprocess = True, use_gpu = False, silent=True, html=False)

        # best_model_list = compare_models(include=['lr', 'knn', 'svm', 'nb', 'ada', 'dt', 'rf', 'lightgbm', 'mlp'], sort = 'Accuracy', n_select=10)
        best_model_list = compare_models(include=['lr', 'knn'], sort = 'Accuracy', n_select=10)
        score_grid = pull()

        prep_pipe = get_config('prep_pipe')
        transformed_df = prep_pipe.transform(test_df)
        
        for best_model in best_model_list:
            modelname = type(best_model).__name__
            pred_df = predict_model(best_model)

            try:
                y_true = pred_df['target']
                y_pred = pred_df['Label']
                y_pred_proba = best_model.predict_proba(transformed_df)

                with open(os.path.join('out_clf_ml', user_id, '{}_{}'.format(modelname, k)), 'wb') as f:
                    pickle.dump([y_true, y_pred, y_pred_proba], f)
                    
            except:
                y_true = pred_df['target']
                y_pred = pred_df['Label']

                with open(os.path.join('out_clf_ml', user_id, '{}_{}'.format(modelname, k)), 'wb') as f:
                    pickle.dump([y_true, y_pred], f)



