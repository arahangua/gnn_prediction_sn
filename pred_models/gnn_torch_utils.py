#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:23:59 2021

@author: th
"""


import torch
from torch.nn import ReLU, Linear, Softmax, SmoothL1Loss, Tanh, LeakyReLU
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool, SGConv, GNNExplainer, SAGEConv, GATConv, FastRGCNConv, GraphConv
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F
import torch_optimizer as optim

import gnn_torch_models

import random
from sklearn.preprocessing import StandardScaler as SS
# torch.set_default_dtype(torch.float)


def standardscaler_transform(sc_feat_pure):
    scaler = SS()
    scaler.fit(sc_feat_pure)
    transformed=scaler.transform(sc_feat_pure)
    
    return transformed, scaler
  
def batch_split(nodes_cp, full_index, ii):
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
    
def make_diag_batch_FC(FCs):
    
    count=0
    for FC in FCs:
        count+=FC.shape[0]
        
    #gen mat
    
    batch_FC = np.zeros((count,count))
    size_log = 0
    for FC in FCs:
        size = FC.shape[0]
        batch_FC[size_log:size_log+size, size_log:size_log+size]=FC
        size_log += size
    
    return batch_FC

def flatten_list_1d(act_ratio):
    ph = np.empty((1,0))
    ph = np.squeeze(ph)
    
    for entry in act_ratio:
        ph = np.concatenate((ph, entry))
        
    return ph

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



def evaluate(out, labels):
    """
    Calculates the accuracy between the prediction and the ground truth.
    :param out: predicted outputs of the explainer
    :param labels: ground truth of the data
    :returns: int accuracy
    """
    acc = torch.mean(torch.square(out-labels))
    return acc  
def evaluate_mae(out, labels):
    """
    Calculates the accuracy between the prediction and the ground truth.
    :param out: predicted outputs of the explainer
    :param labels: ground truth of the data
    :returns: int accuracy
    """
    acc = torch.mean(torch.abs(out-labels))
    return acc  
def evaluate_acc(out, labels):
    """
    Calculates the accuracy between the prediction and the ground truth.
    :param out: predicted outputs of the explainer
    :param labels: ground truth of the data
    :returns: int accuracy
    """
    out_cl = torch.max(out,1)[1]
    lab_cl = torch.max(labels,1)[1]
    diff_sum = torch.sum(torch.abs(out_cl-lab_cl))
    
    acc = 1- (diff_sum/out.shape[0])
    return acc  





def gen_gridparams(dropout_probs, learning_rates, weight_decays, hidden_dims):
    
    fit_param_list = []
    for prob in dropout_probs:
        for rate in learning_rates:
            for decay in weight_decays:
                for hd in hidden_dims:
                    fit_params= dict()
                    fit_params['dropout_prob']=prob
                    fit_params['learning_rate']=rate
                    fit_params['weight_decay']=decay
                    fit_params['hidden_dims']=hd
                    fit_param_list.append(fit_params)
    return fit_param_list
    
            



def run_gridsearch_batch_x(nodes, FCs, target_frs, epoch_n, iter_n, model_string, fit_param_list, device, chip_ids):
    
    fit_result=[]
    for entry in fit_param_list:
        fit_params= dict()
        fit_params['dropout_prob']=entry['dropout_prob']
        fit_params['learning_rate']=entry['learning_rate']
        fit_params['weight_decay']=entry['weight_decay']
        fit_params['hidden_dims']=entry['hidden_dims']
        
        fit_params['fit_result']=run_GNN_batch_x(nodes, FCs, target_frs, epoch_n, iter_n, model_string, fit_params, device, chip_ids, 1)
        fit_result.append(fit_params)    
    
    return fit_result


    
def standard_scale(features,train_idx, validate_idx, test_idx):
    features_wip = np.copy(features)
    
    if(len(features_wip.shape)==1):
        X_train, X_scaler = standardscaler_transform(features_wip[train_idx].reshape(-1,1))
        X_validate = X_scaler.transform(features_wip[validate_idx].reshape(-1,1))
        X_test = X_scaler.transform(features_wip[test_idx].reshape(-1,1))
        features_wip[train_idx] = np.squeeze(X_train)
        features_wip[validate_idx] = np.squeeze(X_validate)
        features_wip[test_idx] = np.squeeze(X_test)
    else:    
        X_train, X_scaler = standardscaler_transform(features_wip[train_idx, :])
        X_validate = X_scaler.transform(features_wip[validate_idx, :])
        X_test = X_scaler.transform(features_wip[test_idx, :])
        features_wip[train_idx, :] = X_train
        features_wip[validate_idx, :] = X_validate
        features_wip[test_idx, :] = X_test
    
    return features_wip
    
    


def make_rgcn_mat(train_FC, device):
    edge_idx = np.array(np.where(train_FC!=0))
    edge_idx = torch.tensor(edge_idx, device= device)
    edge_type = train_FC[np.where(train_FC!=0)]
    types = np.unique(edge_type)
    edge_class = np.squeeze(np.zeros((edge_type.shape[0],1)))
    for jj, typ in enumerate(types):
        idx = np.where(edge_type==typ)[0]
        edge_class[idx]=jj
    edge_weight = torch.tensor(edge_class, device=device).type(torch.LongTensor)
    
    return edge_idx, edge_weight
                


def match_network_param(sage_params_uniq, chip_ids):
    uniq_chip = np.unique(chip_ids)
    uniq_indices=[]
    for uniq_c in uniq_chip:
        indices = np.where(np.array(chip_ids)==uniq_c)[0]
        uniq_indices.append(indices[0])
        
    
    sage_params = dict()
    for k,v in sage_params_uniq.items():
        sage_params[k] = []
    
    # get the sequence straight
    
    seq = np.argsort(uniq_indices) 
    for k,v in sage_params_uniq.items():
        for zz, idx in enumerate(seq):
            st_p=uniq_indices[idx]
            n_same = len(np.where(np.array(chip_ids)==np.array(chip_ids[st_p]))[0])
            for _ in range(n_same):
                sage_params[k].append(sage_params_uniq[k][zz])
    
    return sage_params 




def run_GNN_batch_x(nodes, FCs, target_frs, n_epoch, iter_n, model_string, fit_params_list, device, chip_ids, gridsearch=0):
    # compute GCN assuming same nodes
    
    #seeds
    np.random.seed(42)
    random.seed(42)
    num_features= nodes[0].shape[1]
    
    #number of classes
    if(len(target_frs[0].shape)==1):
         num_classes=1
    else:
         num_classes = target_frs[0].shape[1]
    
    
    per_network=[]
    for ii in range(len(target_frs)):
        train_acc_vec=[]
        train_mae_vec=[]
        model_params_vec=[]
        test_acc_vec=[]
        test_mae_vec=[]

        
        validate_curves_list =[]
        train_curves_list=[]

        # prep x,y 
        target_cp = np.copy(target_frs)
        full_index= np.arange(len(target_frs))
        #get target y first 
        test_y = target_cp[ii]
        # make x 
        nodes_cp = np.copy(nodes)
        # FC
        FC_cp = np.copy(FCs)
        
        #params 
        if(gridsearch==0):
            fit_params = fit_params_list[ii]
        else:
            fit_params = fit_params_list
        
        for iter_ in range(iter_n):
             
            # targets
            test_y = target_cp[ii]
            # val_y = target_cp[val_idx]
            
            #get idx from same chips 
            same_chip = np.where(np.array(chip_ids) == chip_ids[ii])[0]
            
            if(gridsearch==0):
                train_idx=np.setxor1d(full_index, same_chip) # got rid of it
            else:
                train_idx = np.setxor1d(full_index, ii)
                
            train_y = target_cp[train_idx]
            train_y = flatten_list_1d(train_y)
            
            # make x 
            #features (input)
            if(gridsearch==0):
                train_x, test_x= batch_split_x(nodes_cp, full_index, ii, same_chip) #identical function to wp1_data_description, wp1_data class
            else:
                train_x, test_x= batch_split(nodes_cp, full_index, ii)
                
            #stack train and val for scaling 
            
            #scale them
            scaled_x, train_scaler_x=standardscaler_transform(train_x)
            test_x = train_scaler_x.transform(test_x) 
            train_x = train_scaler_x.transform(train_x)
            # val_x = train_scaler_x.transform(val_x)
            
            # scale y
            
            scaled_y, train_scaler_y=standardscaler_transform(train_y.reshape(-1,1))
            train_y = train_scaler_y.transform(train_y.reshape(-1,1))
            test_y = train_scaler_y.transform(test_y.reshape(-1,1))
            # val_y = train_scaler_y.transform(val_y.reshape(-1,1))
            
            # FCs
            train_FC= make_diag_batch_FC(FC_cp[train_idx])
            test_FC = FC_cp[ii]
            # put into cuda 
            train_x = torch.tensor(train_x, device = device, dtype=float)
            train_y = torch.tensor(train_y, device = device, dtype=float)
            test_x = torch.tensor(test_x, device = device, dtype=float)
            test_y = torch.tensor(test_y, device = device, dtype=float)
            
            if(num_classes==1):
                train_y = torch.reshape(train_y, (train_y.shape[0], 1))
                test_y = torch.reshape(test_y, (test_y.shape[0], 1))
                
            edge_idx= dict()
            edge_weight =dict()
            edge_idx['train'] = np.array(np.where(train_FC>0))
            edge_idx['train'] = torch.tensor(edge_idx['train'], device = device)
            edge_weight['train'] = train_FC[np.where(train_FC>0)]
            edge_weight['train'] = torch.tensor(edge_weight['train'], device = device, dtype=float)
                        
            #prep for testing 
           
            edge_idx['test'] = np.array(np.where(test_FC>0))
            edge_idx['test'] = torch.tensor(edge_idx['test'], device = device)
            edge_weight['test'] = test_FC[np.where(test_FC>0)]
            edge_weight['test'] = torch.tensor(edge_weight['test'], device = device,  dtype=float)
            
            
            
            model = gnn_torch_models.return_model(model_string, num_features, num_classes, fit_params['dropout_prob'], fit_params['hidden_dims'])
            model.to(device, dtype=float)            
            
            if('rgcn' in model_string):
                edge_idx= dict()
                edge_weight =dict()
                edge_idx['train'], edge_weight['train'] = make_rgcn_mat(train_FC, device)
                edge_idx['test'], edge_weight['test'] = make_rgcn_mat(test_FC, device)
                edge_idx['train'] = torch.tensor(edge_idx['train'], device= device)
                edge_idx['test'] = torch.tensor(edge_idx['test'], device= device)
                edge_weight['train'] = torch.tensor(edge_weight['train'], device= device,  dtype=float)
                edge_weight['test'] = torch.tensor(edge_weight['test'], device= device,  dtype=float)
                
                # edge_idx['val'], edge_weight['val'] = make_rgcn_mat(val_FC, device)
                
            
            optimizer = torch.optim.Adam(model.parameters(), lr=fit_params['learning_rate'], weight_decay= fit_params['weight_decay'])
          
            if(model_string == 'gcn_class'):
                criterion = torch.nn.CrossEntropyLoss()
            else:
                criterion = torch.nn.MSELoss()
               
            train_acc_curve=[]
            validate_acc_curve=[]
            
            
            #epochs
            if(gridsearch==0):
                n_epoch = fit_params['bs_epoch']
                
            
            for epoch in range(n_epoch):
                
                model.train()
                optimizer.zero_grad()
                out = model.forward(train_x, edge_idx['train'], edge_weight['train']) # forwarding x has 0 for single wfs defected ones
                loss = criterion(out, train_y)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                
                # eval flag
                model.eval()
                
                with torch.no_grad():
                    out=dict()
                    out['train'] = model(train_x, edge_idx['train'], edge_weight['train'])
                    out['test'] = model(test_x, edge_idx['test'], edge_weight['test'])
                    # out['test'] = model(test_x, edge_idx['test'], edge_weight['test'])
                    
                    # Evaluate train
                    mse=dict()
                    mae=dict()
                    mse['train'] = evaluate(out['train'], train_y)
                    mae['train'] = evaluate_mae(out['train'], train_y)
                    
                    mse['test'] = evaluate(out['test'], test_y)
                    mae['test'] = evaluate_mae(out['test'], test_y)
                    
                
                
                if(epoch% 50==0):
                    print(f"Epoch: {epoch}, train_acc: {mse['train']:.4f}, validate_acc : {mse['test']:.4f}, LR : {optimizer.param_groups[0]['lr']:.8f}")
                    train_acc_curve.append(mse['train'].cpu().numpy())
                    validate_acc_curve.append(mse['test'].cpu().numpy())
                          
            
                        
            # for each iter
            train_acc_vec.append(mse['train'].cpu().numpy())
            train_mae_vec.append(mae['train'].cpu().numpy())
           
            validate_curves_list.append(np.array(validate_acc_curve))
            train_curves_list.append(np.array(train_acc_curve))
            
            model_dict=dict()
            
            for k,v in model.state_dict().items():
                model_dict[k] =v.cpu()
            
            if(gridsearch==0):
                model_params_vec.append(model_dict)
            
            
           # test
            with torch.no_grad():
                out['test'] = model(test_x, edge_idx['test'], edge_weight['test'])
                mse['test'] = evaluate(out['test'], test_y)
                mae['test'] = evaluate_mae(out['test'], test_y)
                print(f"iteration: {iter_}, test_acc: {mse['test']:.4f}")
            
            test_acc_vec.append(mse['test'].cpu().numpy())
            test_mae_vec.append(mae['test'].cpu().numpy())
         
        result = dict()
        result['mse_train']=np.array(train_acc_vec)
        result['mae_train']=np.array(train_mae_vec)
        
        result['mse_test']= np.array(test_acc_vec)
        result['mae_test'] = np.array(test_mae_vec)
       
        result['train_curve']=train_curves_list
        result['validate_curve']=validate_curves_list
        per_network.append(result)
    return per_network





def simple_forward_model(model, input_vec, adj_mat, cuda, gpu_id):
    
    x = torch.tensor(input_vec)
  #  lab_out = torch.tensor(target_vec)
   # lab_out = torch.reshape(lab_out, (adj_mat.shape[0], 1))
    edge_idx = np.array(np.where(adj_mat>0))
    edge_idx = torch.tensor(edge_idx)
    edge_weight = adj_mat[np.where(adj_mat>0)]
    edge_weight = torch.tensor(edge_weight)
    
    
    
    if(cuda):
   #     lab_out=lab_out.cuda(gpu_id)
        x = x.cuda(gpu_id)
        edge_idx = edge_idx.cuda(gpu_id)
        edge_weight=edge_weight.cuda(gpu_id)
        model = model.cuda(gpu_id)
    
    
    with torch.no_grad():
        out = model.forward(x, edge_idx, edge_weight)
    
    return out.cpu().detach().numpy()

