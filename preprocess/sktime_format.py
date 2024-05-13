import os
import sys
import time
import math
import copy
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime, timedelta

import warnings
from pandas.core.common import SettingWithCopyWarning

from sktime.datasets import load_UCR_UEA_dataset
from sktime.utils.data_processing import from_2d_array_to_nested

# check path
if not os.path.exists('sktime'):
    os.mkdir('sktime')

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
    'user21': 'user21-25', 'user22': 'user21-25', 'user23': 'user21-25', 'user24': 'user21-25', # 'user25': 'user21-25', 
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

user_list = list(path_dict.keys())

for user_id in user_list:
    print(user_id)
    try:
        with open('preprocess/{}'.format(user_id), 'rb') as f:
            all_features = pickle.load(f)
    except:
        continue
        
    if len(all_features) < 1000:
        continue

    # transformation
    X_data = []
    y_data = []

    for feature in tqdm(all_features):
        y1, y2, y3 = label_dict[feature[0]['action']], feature[0]['emotionPositive'], feature[0]['emotionTension']

        # def sampling function
        def sampling(ff, ff_len=60):
            ff.index = pd.to_timedelta(ff.index, unit='s')
            bin_len = (ff.index[-1]-ff.index[0])/ff_len
            try:
                ff = ff.resample('{:0.5f}S'.format(bin_len.total_seconds())).mean().fillna(method='ffill')[:ff_len]
            except:
                ff = ff.resample('{:0.4f}S'.format(bin_len.total_seconds())).mean().fillna(method='ffill')[:ff_len]
            return ff
        
        # preprocessing to same length
        seq_len = 60
        feature_all = []
        for i in range(3,12):
            ff = feature[i]
            if len(ff) < 60:
                ff = sampling(ff, ff_len=60)

            bin_len = int(len(ff)/60)
            feature_ds = ff.to_numpy().reshape(ff.shape[1],-1,bin_len).mean(axis=2)
            feature_all.append(feature_ds)
        feature_all = np.vstack(feature_all)
        sktime_feature = from_2d_array_to_nested(feature_all).T

        X_data.append(sktime_feature)
        y_data.append([y1, y2, y3])

    X_data = pd.concat(X_data)
    y_data = np.array(y_data)
    
    with open('sktime/{}'.format(user_id), 'wb') as f:
        pickle.dump([X_data, y_data], f)