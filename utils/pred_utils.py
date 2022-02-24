#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 20:54:01 2022

@author: th
"""
import numpy as np

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


def batch_split(self, nodes_cp, full_index, ii):
    
    test_x = nodes_cp[ii]
    train_idx=np.setxor1d(full_index, ii)
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



def batch_split_x(self, nodes_cp, full_index, ii, chip_ids):
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