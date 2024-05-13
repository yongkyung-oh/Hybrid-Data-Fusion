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

# check path
if not os.path.exists('preprocess'):
    os.mkdir('preprocess')

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

user_list = list(path_dict.keys())

for user_id in user_list:
    print(user_id)
    
    selected_user_info = df_user_info.loc[[user_id]]
    selected_user_sleep = df_user_sleep.loc[[user_id]]
    selected_user_survey = df_user_survey.loc[[user_id]]

    user_path = path_dict[user_id]
    data_path = '/media/usr/HDD/Data/Lifelog/'
    path_list = os.listdir(os.path.join(data_path, user_path, user_id))
    path_list = sorted(path_list)

    label_all = []
    for path in path_list:
        label_df = pd.read_csv(os.path.join(data_path, user_path, user_id, path, '{}_label.csv'.format(path)), index_col=0)
        label_all.append(label_df)
    label_all_df = pd.concat(label_all)

    def sampling(ff, ff_len=60):
        ff.index = pd.to_timedelta(ff.index, unit='s')
        bin_len = (ff.index[-1]-ff.index[0])/(ff_len+1)
        try:
            ff = ff.resample('{:0.5f}S'.format(bin_len.total_seconds())).mean().fillna(method='ffill')[:ff_len]
        except:
            ff = ff.resample('{:0.4f}S'.format(bin_len.total_seconds())).mean().fillna(method='ffill')[:ff_len]

        if len(ff) == ff_len:
            return ff
        else:
            raise ValueError()
    
    # get features
    label_selected_list = []
    f1_list = [] # e4Acc # 1920, 3
    f2_list = [] # e4Bvp # 3840, 1
    f3_list = [] # e4Eda # 240, 1
    f4_list = [] # e4Hr # 60, 1
    f5_list = [] # e4Temp # 240, 1
    f6_list = [] # mAcc # 1800, 3
    f7_list = [] # mGps # 20, 3
    f8_list = [] # mGyr # 1800, 6
    f9_list = [] # mMag # 1800, 3

    for path in tqdm(path_list):
        label_df = pd.read_csv(os.path.join(data_path, user_path, user_id, path, '{}_label.csv'.format(path)), index_col=0)
        for idx in label_df.index:
            idx = str(int(idx))
            label = label_df.loc[[int(idx)]].iloc[0]

            # select feature 
            try:
                f1 = pd.read_csv(os.path.join(data_path, user_path, user_id, path, 'e4Acc', idx+'.csv'), index_col=0)
                f2 = pd.read_csv(os.path.join(data_path, user_path, user_id, path, 'e4Bvp', idx+'.csv'), index_col=0)
                f3 = pd.read_csv(os.path.join(data_path, user_path, user_id, path, 'e4Eda', idx+'.csv'), index_col=0)
                f4 = pd.read_csv(os.path.join(data_path, user_path, user_id, path, 'e4Hr', idx+'.csv'), index_col=0)
                f5 = pd.read_csv(os.path.join(data_path, user_path, user_id, path, 'e4Temp', idx+'.csv'), index_col=0)
                f6 = pd.read_csv(os.path.join(data_path, user_path, user_id, path, 'mAcc', idx+'.csv'), index_col=0)
                f7 = pd.read_csv(os.path.join(data_path, user_path, user_id, path, 'mGps', idx+'.csv'), index_col=0)
                f8 = pd.read_csv(os.path.join(data_path, user_path, user_id, path, 'mGyr', idx+'.csv'), index_col=0)
                f9 = pd.read_csv(os.path.join(data_path, user_path, user_id, path, 'mMag', idx+'.csv'), index_col=0)

                # check at least 50% of data
                if (len(f1) < 1920/2): continue
                if (len(f2) < 3840/2): continue
                if (len(f3) < 240/2): continue
                if (len(f4) < 60/2): continue
                if (len(f5) < 240/2): continue
                if (len(f6) < 1800/2): continue
                if (len(f7) < 20/2): continue
                if (len(f8) < 1800/2): continue
                if (len(f9) < 1800/2): continue

                # data sampling
                f1 = sampling(f1, 1920)
                f2 = sampling(f2, 3840)
                f3 = sampling(f3, 240)
                f4 = sampling(f4, 60)
                f5 = sampling(f5, 240)
                f6 = sampling(f6, 1800)
                f7 = sampling(f7, 20)
                f8 = sampling(f8, 1800)
                f9 = sampling(f9, 1800)

                f1_list.append(f1)
                f2_list.append(f2)
                f3_list.append(f3)
                f4_list.append(f4)
                f5_list.append(f5)
                f6_list.append(f6)
                f7_list.append(f7)
                f8_list.append(f8)
                f9_list.append(f9)

                label_selected_list.append(label)
            except:
                pass

    try:
        f1_all = pd.concat(f1_list)
        f2_all = pd.concat(f2_list)
        f3_all = pd.concat(f3_list)
        f4_all = pd.concat(f4_list)
        f5_all = pd.concat(f5_list)
        f6_all = pd.concat(f6_list)
        f7_all = pd.concat(f7_list)
        f8_all = pd.concat(f8_list)
        f9_all = pd.concat(f9_list)
    except:
        continue

    label_selected_df = pd.concat(label_selected_list, axis=1).T

    # combine multi-modal features
    all_features = []
    for i in tqdm(range(len(label_selected_df))):
        label = label_selected_df.iloc[i]
        user_features = selected_user_info.copy()

        # user info
        user_features['gender'] = user_features['gender'].replace({'M':0, 'F':1})
        user_features['handed'] = user_features['handed'].replace({'Right':0, 'Left':1})
        user_features = user_features.iloc[0].to_numpy()[:5]

        # daily sleep feature
        ts = float(label.name)
        dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")

        try:
            slp_features = selected_user_sleep.loc[selected_user_sleep['date']==dt]
            slp_features = slp_features.iloc[0, 5:].to_numpy().astype(float)
        except:
            continue

        # combined features 
        all_features.append([
            label, user_features, slp_features, 
            f1_list[i], f2_list[i], f3_list[i], f4_list[i], f5_list[i], f6_list[i], f7_list[i], f8_list[i], f9_list[i]
        ])    

    with open('preprocess/{}'.format(user_id), 'wb') as f:
        pickle.dump(all_features, f)