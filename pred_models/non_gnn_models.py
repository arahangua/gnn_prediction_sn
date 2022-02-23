#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 20:49:36 2022

@author: th
"""

import numpy as np

# import ray

import random
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler as SS


def batch_split_x(nodes_cp, full_index, ii, chip_ids):
    nodes_cp = np.array(nodes_cp)
    test_x = nodes_cp[ii]
    train_idx=np.setxor1d(full_index, chip_ids)
    train_x = nodes_cp[train_idx]
    if(len(train_x[0].shape)==1):
        train_concat = flatten_list_1d(train_x)
    else:
        train_concat = []
        for jj, x in enumerate(train_x):
            if(jj==0):
                train_concat = x
            else:
                train_concat= np.vstack((train_concat, x))
                
    return train_concat, test_x




def flatten_list_1d(act_ratio):
    ph = np.empty((1,0))
    ph = np.squeeze(ph)
    
    for entry in act_ratio:
        ph = np.concatenate((ph, entry))
        
    return ph


def standardscaler_transform(sc_feat_pure):
    scaler = SS()
    scaler.fit(sc_feat_pure)
    transformed=scaler.transform(sc_feat_pure)
    
    return transformed, scaler


def average_mse_batch_x(target_frs, y_scale, chip_ids):
    mse_vec = []
    mse_train= []
    just_ave = []
    
    mae_vec = []
    mae_train= []
    just_ave_mae = []
    
    for ii in range(len(target_frs)):
        target_cp = np.copy(target_frs)
        full_index= np.arange(len(target_frs))

        
        test_x = target_cp[ii]
        #also take out configs belonging to the same chip 
        same_chip = np.where(np.array(chip_ids) == chip_ids[ii])[0]
        train_idx=np.setxor1d(full_index, same_chip)
        train_x = target_cp[train_idx]
        
        # concat all train set
        train_x = flatten_list_1d(train_x)
 
        #standardize
        if(y_scale):
            train_x, train_scaler_x= standardscaler_transform(train_x.reshape(-1,1))
            test_x = train_scaler_x.transform(test_x.reshape(-1,1)) 
        
     
        
        mean_train = np.mean(train_x)
        mse_loss = np.mean((test_x-mean_train)**2)
        mse_loss_tr = np.mean((train_x-mean_train)**2)
        mse_vec.append(mse_loss)
        mse_train.append(mse_loss_tr)
        mean_test = np.mean(test_x)
        mse_pure = np.mean(np.square(test_x-mean_test))
        just_ave.append(mse_pure)
        
        #mae
        mae_loss = np.mean(np.abs(test_x-mean_train))
        mae_loss_tr = np.mean(np.abs(train_x-mean_train))
        mae_vec.append(mae_loss)
        mae_train.append(mae_loss_tr)
        
        mean_test = np.mean(test_x)
        mae_pure = np.mean(np.abs(test_x-mean_test))
        just_ave_mae.append(mae_pure)
        
    ave_result = dict()
    ave_result['mse_test']= np.array(mse_vec)
    ave_result['mse_train']= np.array(mse_train)
    ave_result['mae_test']= np.array(mae_vec)
    ave_result['mae_train']= np.array(mae_train)
    
    
    return ave_result


def linear_reg_batch_x(nodes, target_frs, iter_n, y_scale, chip_ids):
    np.random.seed(42)
    random.seed(42)
    full_index= np.arange(len(target_frs))
    per_network = []
    for ii in range(len(target_frs)):
        
        ls_vec=[]
        lin_coef_vec=[]
        mse_vec=[]
        mae_vec=[]
        y_pred_vec = []
        ls_vec_t=[]
        mse_vec_t=[]
        mae_vec_t=[]
        #y_pred_vec_t = []
        #get target y first 
        target_cp = np.copy(target_frs)
        full_index= np.arange(len(target_frs))
        test_y = target_cp[ii]
        #get idx from same chips 
        same_chip = np.where(np.array(chip_ids) == chip_ids[ii])[0]
        
        train_idx=np.setxor1d(full_index, same_chip) # got rid of it
        
        train_y = target_cp[train_idx]
        train_y = flatten_list_1d(train_y)
        
        # make x 
        nodes_cp = np.copy(nodes)
        train_x, test_x = batch_split_x(nodes_cp, full_index, ii, same_chip)
        
        train_x, train_scaler_x= standardscaler_transform(train_x)
        test_x = train_scaler_x.transform(test_x) 
        
        if(y_scale):
            train_y, train_scaler_y=standardscaler_transform(train_y.reshape(-1,1))
            test_y = train_scaler_y.transform(test_y.reshape(-1,1)) 
            
        
        for iter_ in range(iter_n):
             
            reg = LinearRegression().fit(train_x, train_y)
            linear_score = reg.score(train_x, train_y)
            linear_coef = reg.coef_
            y_pred=reg.predict(train_x)
            mseloss = np.mean(((train_y - y_pred) ** 2))
            maeloss = np.mean(np.abs(train_y-y_pred))
               
            ls_vec.append(linear_score)
            lin_coef_vec.append(linear_coef)
            mse_vec.append(mseloss)
            y_pred_vec.append(y_pred)
            mae_vec.append(maeloss)
            
            y_pred = reg.predict(test_x)
            mseloss= np.mean(((test_y - y_pred) ** 2))
            maeloss = np.mean(np.abs(test_y-y_pred))
            
            ls_vec_t.append(reg.score(test_x, test_y))
            mse_vec_t.append(mseloss)
            mae_vec_t.append(maeloss)
            # y_pred_vec_t.append(y_pred)
             

  
  
        lin_result = dict()
        lin_result['R-sq']=np.array(ls_vec)
        lin_result['slope_coef']=np.array(lin_coef_vec)
        lin_result['mse_train']=np.array(mse_vec)
        lin_result['mae_train'] = np.array(mae_vec)
        lin_result['pred']=y_pred_vec
        lin_result['R-sq test']= np.array(ls_vec_t)
        lin_result['mse_test'] = np.array(mse_vec_t)
        lin_result['mae_test'] = np.array(mae_vec_t)
        per_network.append(lin_result)
        
    return per_network


def rf_reg_batch_x(nodes, target_frs, iter_n, y_scale, chip_ids, params):
    np.random.seed(42)
    random.seed(42)
    full_index= np.arange(len(target_frs))
    per_network = []
    for ii in range(len(target_frs)):
        
        ls_vec = []
        mse_vec= []
        mae_vec=[]
        y_pred_vec=[]
        feat_imp_vec = []
        mse_test_vec=[]
        mae_test_vec=[]
        #y_pred_vec_t = []
        #get target y first 
        
        
         #get target y first 
        target_cp = np.copy(target_frs)
        full_index= np.arange(len(target_frs))
        test_y = target_cp[ii]
        #get idx from same chips 
        same_chip = np.where(np.array(chip_ids) == chip_ids[ii])[0]
        train_idx=np.setxor1d(full_index, same_chip) # got rid of it
        
        train_y = target_cp[train_idx]
        train_y = flatten_list_1d(train_y)
        
        # make x 
        nodes_cp = np.copy(nodes)
        train_x, test_x = batch_split_x(nodes_cp, full_index, ii, same_chip)
        train_x, train_scaler_x=standardscaler_transform(train_x)
        test_x = train_scaler_x.transform(test_x) 
        
   
        
        if(y_scale):
            train_y, train_scaler_y=standardscaler_transform(train_y.reshape(-1,1))
            test_y = train_scaler_y.transform(test_y.reshape(-1,1)) 
            train_y = np.squeeze(train_y)
            test_y = np.squeeze(test_y)
        
        for iter_ in range(iter_n):
             
            if(type(params)==bool):
                reg = RandomForestRegressor(
                                             n_estimators = 200,
                                              max_features = 'sqrt', 
                                      min_samples_leaf = 5, 
                                      min_samples_split = 2,
                                            )
            else:
                reg = RandomForestRegressor(n_estimators = params[ii]['rf__n_estimators'], 
                                     # max_depth= params['max_depth'], 
                                      max_features = params[ii]['rf__max_features'], 
                                     min_samples_leaf = params[ii]['rf__min_samples_leaf'], 
                                     min_samples_split = params[ii]['rf__min_samples_split'],
                                     )
            
            reg.fit(train_x, train_y)
            y_pred = reg.predict(train_x)
            mseloss = np.mean(((train_y - y_pred) ** 2))
            maeloss = np.mean(np.abs(train_y-y_pred))
            y_pred = reg.predict(test_x)
            mseloss_test = np.mean(((test_y - y_pred) ** 2))
            maeloss_test = np.mean(np.abs(test_y-y_pred))        
            feat_imp = reg.feature_importances_
            
            ls_vec.append(reg.score(train_x, train_y))
            mse_vec.append(mseloss)
            mae_vec.append(maeloss)
            y_pred_vec.append(y_pred)
            feat_imp_vec.append(feat_imp)
            mse_test_vec.append(mseloss_test)
            mae_test_vec.append(maeloss_test)
            

   
#load_lr_model =pickle.load(open(filename, 'rb'))
  
        rf_result = dict()
        rf_result['reg_score']=np.array(ls_vec)
        rf_result['mse_train']=np.array(mse_vec)
        rf_result['y_pred']=np.array(y_pred_vec)
        rf_result['feat_importance']=feat_imp_vec
        rf_result['mse_test'] = np.array(mse_test_vec)
        rf_result['mae_train'] = np.array(mae_vec)
        rf_result['mae_test'] = np.array(mae_test_vec)
        per_network.append(rf_result)
        
    return per_network