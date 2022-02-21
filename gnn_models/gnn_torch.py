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

sys.path.append('/home/th/bsse_home/work_directory/wp1/python/pgexplainer')
sys.path.append('/home/takim/work_directory/wp1/python/pgexplainer')

from ExplanationEvaluation.explainers.PGExplainer import PGExplainer
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


from sklearn.model_selection import train_test_split
import neural_assembly_utils
import neural_assembly_utils2
import neural_assembly_utils3

import random

torch.set_default_dtype(torch.float64)



class RGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_edge_types, dropout_prob, hidden):
        super(RGCN, self).__init__()
        self.embedding_size = hidden * 3
        self.conv1 = FastRGCNConv(num_features, hidden, num_edge_types)
        self.relu1 = ReLU()
        self.conv2 = FastRGCNConv(hidden, hidden, num_edge_types)
        self.relu2 = ReLU()
        self.conv3 = FastRGCNConv(hidden, hidden, num_edge_types)
        self.relu3 = ReLU()
        self.lin = Linear(3*hidden, num_classes)
        self.dropout_prob = dropout_prob
        
        
    def forward(self, x, edge_index, edge_type):
        input_lin = self.embedding(x, edge_index, edge_type)
        final = self.lin(input_lin)
        return final
    def embedding(self, x, edge_index, edge_type):
       # if edge_weights is None:
       #     edge_weights = torch.ones(edge_index.size(1))
       stack = []

       out1 = self.conv1(x, edge_index, edge_type)
       out1 = F.normalize(out1, p=2, dim=0)  # this is not used in PGExplainer
       out1 = F.dropout(out1, training=self.training, p=self.dropout_prob)
       out1 = self.relu1(out1)
       
       stack.append(out1)

       out2 = self.conv2(out1, edge_index, edge_type)
       out2 = torch.nn.functional.normalize(out2, p=2, dim=0)  # this is not used in PGExplainer
       out2= F.dropout(out2,training=self.training, p=self.dropout_prob)
       out2 = self.relu2(out2)
       stack.append(out2)

       out3 = self.conv3(out2, edge_index, edge_type)
       out3 = torch.nn.functional.normalize(out3, p=2, dim=0)  # this is not used in PGExplainer
       out3 = F.dropout(out3,training=self.training, p=self.dropout_prob)
       out3 = self.relu3(out3)
       stack.append(out3)

       input_lin = torch.cat(stack, dim=1)

       return input_lin   
   
    
class sl_RGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_edge_types, dropout_prob, hidden):
        super(sl_RGCN, self).__init__()
        self.embedding_size = hidden * 1
        self.conv1 = FastRGCNConv(num_features, hidden, num_edge_types)
        self.relu1 = ReLU()
        self.lin = Linear(1*hidden, num_classes)
        self.dropout_prob = dropout_prob
        
        
    def forward(self, x, edge_index, edge_type):
        input_lin = self.embedding(x, edge_index, edge_type)
        final = self.lin(input_lin)
        return final
    def embedding(self, x, edge_index, edge_type):
        # if edge_weights is None:
        #     edge_weights = torch.ones(edge_index.size(1))
        stack = []

        out1 = self.conv1(x, edge_index, edge_type)
        out1 = F.normalize(out1, p=2, dim=0)  # this is not used in PGExplainer
        out1 = F.dropout(out1, training=self.training, p=self.dropout_prob)
        out1 = self.relu1(out1)
        stack.append(out1)

        input_lin = torch.cat(stack, dim=1)

        return input_lin
    
class RGCN2(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_edge_types, dropout_prob, hidden):
        super(RGCN2, self).__init__()
        self.embedding_size = hidden * 2
        self.conv1 = FastRGCNConv(num_features, hidden, num_edge_types)
        self.relu1 = ReLU()
        self.conv2 = FastRGCNConv(hidden, hidden, num_edge_types)
        self.relu2 = ReLU()
        self.lin = Linear(2*hidden, num_classes)
        self.dropout_prob = dropout_prob
        
        
    def forward(self, x, edge_index, edge_type):
        input_lin = self.embedding(x, edge_index, edge_type)
        final = self.lin(input_lin)
        return final
    def embedding(self, x, edge_index, edge_type):
        # if edge_weights is None:
        #     edge_weights = torch.ones(edge_index.size(1))
        stack = []

        out1 = self.conv1(x, edge_index, edge_type)
        out1 = F.normalize(out1, p=2, dim=0)  # this is not used in PGExplainer
        out1 = F.dropout(out1, training=self.training, p=self.dropout_prob)
        out1 = self.relu1(out1)
        stack.append(out1)
        
        out2 = self.conv2(out1, edge_index, edge_type)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=0)  # this is not used in PGExplainer
        out2= F.dropout(out2,training=self.training, p=self.dropout_prob)
        out2 = self.relu2(out2)
        stack.append(out2)
         

        input_lin = torch.cat(stack, dim=1)

        return input_lin
  
        
        

  

 
    

class Graphsage(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    """
    def __init__(self, num_features, num_classes, dropout_prob, aggre, hidden):
        super().__init__()
        self.embedding_size = hidden * 3
        self.conv1 = GraphConv(num_features, hidden, aggre)
        self.relu1 = ReLU()
        self.conv2 = GraphConv(hidden, hidden, aggre)
        self.relu2 = ReLU()
        self.conv3 = GraphConv(hidden, hidden, aggre)
        self.relu3 = ReLU()
        self.lin = Linear(3*hidden, num_classes)
        self.dropout_prob =  dropout_prob
        
    def forward(self, x, edge_index, edge_weights):
        input_lin = self.embedding(x, edge_index, edge_weights)
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index, edge_weights):
        # if edge_weights is None:
        #     edge_weights = torch.ones(edge_index.size(1))
        stack = []
        
        
        # out1 = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-8)  # this is not used in PGExplainer
        out1 = self.conv1(x, edge_index,edge_weights)
        out1 = F.normalize(out1, p=2, dim=0)  # this is not used in PGExplainer
        out1 = F.dropout(out1, training=self.training, p=self.dropout_prob)
        out1 = self.relu1(out1)
        stack.append(out1)
        # out1 = self.tanh1(out1)
        
        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = F.normalize(out2, p=2, dim=0)  # this is not used in PGExplainer
        out2 = F.dropout(out2, training=self.training, p=self.dropout_prob)
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = F.normalize(out3, p=2, dim=0)  # this is not used in PGExplainer
        out3 = F.dropout(out3, training=self.training, p=self.dropout_prob)
        out3 = self.relu3(out3)
        stack.append(out3)

        input_lin = torch.cat(stack, dim=1)

        return input_lin
    

class Graphsage2(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    """
    def __init__(self, num_features, num_classes, dropout_prob, aggre, hidden):
        super().__init__()
        self.embedding_size = hidden * 2
        self.conv1 = GraphConv(num_features, hidden, aggre)
        self.relu1 = ReLU()
        self.conv2 = GraphConv(hidden, hidden, aggre)
        self.relu2 = ReLU()
        self.lin = Linear(2*hidden, num_classes)
        self.dropout_prob =  dropout_prob
        
    def forward(self, x, edge_index, edge_weights):
        input_lin = self.embedding(x, edge_index, edge_weights)
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index, edge_weights):
        # if edge_weights is None:
        #     edge_weights = torch.ones(edge_index.size(1))
        stack = []
        
        # out1 = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-8)  # this is not used in PGExplainer
        out1 = self.conv1(x, edge_index,edge_weights)
        out1 = F.normalize(out1, p=2, dim=0)  # this is not used in PGExplainer
        out1 = F.dropout(out1, training=self.training, p=self.dropout_prob)
        out1 = self.relu1(out1)
        # out1 = self.tanh1(out1)
        stack.append(out1)
        
        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = F.normalize(out2, p=2, dim=0)  # this is not used in PGExplainer
        out2 = F.dropout(out2, training=self.training, p=self.dropout_prob)
        out2 = self.relu2(out2)
        # out2 = self.tanh2(out2)
        stack.append(out2)

        input_lin = torch.cat(stack, dim=1)

        return input_lin


class Graphsage1(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    """
    def __init__(self, num_features, num_classes, dropout_prob, aggre, hidden):
        super().__init__()
        self.embedding_size = hidden * 1
        self.conv1 = GraphConv(num_features, hidden, aggre)
        self.relu1 = ReLU()
        self.lin = Linear(1*hidden, num_classes)
        self.dropout_prob =  dropout_prob
        
    def forward(self, x, edge_index, edge_weights):
        input_lin = self.embedding(x, edge_index, edge_weights)
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index, edge_weights):
        # if edge_weights is None:
        #     edge_weights = torch.ones(edge_index.size(1))
        stack = []
        
        # out1 = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-8)  # this is not used in PGExplainer
        out1 = self.conv1(x, edge_index,edge_weights)
        out1 = F.normalize(out1, p=2, dim=0)  # this is not used in PGExplainer
        out1 = F.dropout(out1, training=self.training, p=self.dropout_prob)
        out1 = self.relu1(out1)
        stack.append(out1)
        
        input_lin = torch.cat(stack, dim=1)

        return input_lin
    

class NodeGCN(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    """
    def __init__(self, num_features, num_classes, dropout_prob, hidden):
        super(NodeGCN, self).__init__()
        self.embedding_size = hidden * 3
        self.conv1 = GCNConv(num_features, hidden)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(hidden, hidden)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(hidden, hidden)
        self.relu3 = ReLU()
        self.lin = Linear(3*hidden, num_classes)
        self.dropout_prob = dropout_prob

    def forward(self, x, edge_index, edge_weights):
        input_lin = self.embedding(x, edge_index, edge_weights)
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index, edge_weights):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))
        stack = []

        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = F.normalize(out1, p=2, dim=0)  # this is not used in PGExplainer
        out1 = F.dropout(out1, training=self.training, p=self.dropout_prob)
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = F.normalize(out2, p=2, dim=0)  # this is not used in PGExplainer
        out2 = F.dropout(out2, training=self.training, p=self.dropout_prob)
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = F.normalize(out3, p=2, dim=0)  # this is not used in PGExplainer
        out3 = F.dropout(out3, training=self.training, p=self.dropout_prob)
        out3 = self.relu3(out3)
        stack.append(out3)

        input_lin = torch.cat(stack, dim=1)

        return input_lin
    
    
class NodeGCN2(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    """
    def __init__(self, num_features, num_classes, dropout_prob, hidden):
        super(NodeGCN2, self).__init__()
        self.embedding_size =hidden* 2
        self.conv1 = GCNConv(num_features, hidden)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(hidden, hidden)
        self.relu2 = ReLU()
        self.lin = Linear(2*hidden, num_classes)
        self.dropout_prob = dropout_prob

    def forward(self, x, edge_index, edge_weights):
        input_lin = self.embedding(x, edge_index, edge_weights)
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index, edge_weights):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))
        stack = []

        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = F.normalize(out1, p=2, dim=0) 
        out1 = F.dropout(out1, training=self.training, p=self.dropout_prob)
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = F.normalize(out2, p=2, dim=0)  
        out2 = F.dropout(out2, training=self.training, p=self.dropout_prob)
        out2 = self.relu2(out2)
        stack.append(out2)

        input_lin = torch.cat(stack, dim=1)

        return input_lin
   

    

class SlGCN(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    """
    def __init__(self, num_features, num_classes, dropout_prob, hidden):
        super(SlGCN, self).__init__()
        self.embedding_size = hidden
        self.conv1 = GCNConv(num_features, hidden)
        self.relu1 = ReLU()
        self.lin = Linear(hidden, num_classes)
        self.dropout_prob = dropout_prob

    def forward(self, x, edge_index, edge_weights):
        input_lin = self.embedding(x, edge_index, edge_weights)
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index, edge_weights):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))
        stack = []

        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = F.normalize(out1, p=2, dim=0)  # this is not used in PGExplainer
        out1 = F.dropout(out1, training=self.training, p=self.dropout_prob)
        out1 = self.relu1(out1)
        stack.append(out1)

        input_lin = torch.cat(stack, dim=1)

        return input_lin


def batch_split_x(nodes_cp, full_index, ii, chip_ids):
    nodes_cp = np.array(nodes_cp)
    test_x = nodes_cp[ii]
    train_idx=np.setxor1d(full_index, chip_ids)
    train_x = nodes_cp[train_idx]
    if(len(train_x[0].shape)==1):
        train_concat = neural_assembly_utils.flatten_list_1d(train_x)
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



def return_model(model_string, num_features, num_classes, dropout_prob, hidden):
    if(model_string=='gcn'):
        model= NodeGCN(num_features,1, dropout_prob, hidden)
    if(model_string == 'gcn2'):
        model = NodeGCN2(num_features, 1, dropout_prob, hidden)
    if(model_string=='sl_gcn'):
        model = SlGCN(num_features,1, dropout_prob, hidden)
    if(model_string=='sage_max'):
        model = Graphsage(num_features,1, dropout_prob, 'max', hidden)
    if(model_string=='sage_add'):
        model = Graphsage(num_features,1, dropout_prob, 'add', hidden)
    if(model_string=='sage_mean'):
        model = Graphsage(num_features,1, dropout_prob, 'mean', hidden)
        
    if(model_string=='sage2_max'):
        model = Graphsage2(num_features,1, dropout_prob, 'max', hidden)
    if(model_string=='sage2_add'):
        model = Graphsage2(num_features,1, dropout_prob, 'add', hidden)
    if(model_string=='sage2_mean'):
        model = Graphsage2(num_features,1, dropout_prob, 'mean', hidden)
        
        
    if(model_string=='sage1_max'):
        model = Graphsage1(num_features,1, dropout_prob, 'max', hidden)
    if(model_string=='sage1_add'):
        model = Graphsage1(num_features,1, dropout_prob, 'add', hidden)
    if(model_string=='sage1_mean'):
        model = Graphsage1(num_features,1, dropout_prob, 'mean', hidden)
        
        
    if(model_string=='rgcn'):
        model = RGCN(num_features, 1, 2, dropout_prob, hidden)
    if(model_string =='rgcn2'):
        model = RGCN2(num_features, 1, 2, dropout_prob, hidden)
    if(model_string == 'sl_rgcn'):
        model = sl_RGCN(num_features, 1, 2, dropout_prob, hidden)
        
    return model
    

def run_GNN(features, adj_mat, fr_vec, epoch_n, cuda, iteration, gpu_id, valid_mask, cross_validate, x_ratio, random_seeds, model_string):
    # compute GCN assuming same nodes 
    num_features= features.shape[1]
    


    train_acc_vec=[]
    train_mae_vec=[]
    model_params_vec=[]
    test_acc_vec=[]
    test_mae_vec=[]
    flag_vec=[]
    
    valid_mask_use = np.copy(valid_mask) #backup
    valid_mask_bk = np.copy(valid_mask)
    
    for iter_ in range(iteration):
        
        if(cross_validate):
            #valid_idx = np.where(valid_mask==True)[0]
            valid_idx = np.arange(len(fr_vec))
            
            np.random.seed(random_seeds[iter_])
            test_idx=np.random.choice(valid_idx, int(features.shape[0]*x_ratio), replace=False)
            train_idx = np.setxor1d(valid_idx, test_idx)
            
            valid_mask_use = train_idx #getting loss updates only from these
            
        else:
            valid_mask_use = valid_mask_bk
            
        x = torch.tensor(features)
        lab_out = torch.tensor(fr_vec)
        lab_out = torch.reshape(lab_out, (features.shape[0], 1))
   
        
        edge_idx = np.array(np.where(adj_mat>0))
        edge_idx = torch.tensor(edge_idx)
        edge_weight = adj_mat[np.where(adj_mat>0)]
        edge_weight = torch.tensor(edge_weight)
                    
        
        #model = wp1_gnn_torch.NodeGCN(1,1)
        model = return_model(model_string, num_features, 0.1)
        #model = NodeGCN(num_features,1)
        
        if(model_string=='rgcn'):
            edge_idx = np.array(np.where(adj_mat!=0))
            edge_idx = torch.tensor(edge_idx)
            edge_type = adj_mat[np.where(adj_mat!=0)]
            types = np.unique(edge_type)
            edge_class = np.squeeze(np.zeros((edge_type.shape[0],1)))
            for jj, typ in enumerate(types):
                idx = np.where(edge_type==typ)[0]
                edge_class[idx]=jj
            edge_weight = torch.tensor(edge_class).type(torch.LongTensor)
            
        
        if(cuda):
            if(gpu_id==0):
                lab_out=lab_out.cuda()
                x = x.cuda()
                edge_idx = edge_idx.cuda()
                edge_weight=edge_weight.cuda()
                model = model.cuda()
            else:
                lab_out=lab_out.cuda(gpu_id)
                x = x.cuda(gpu_id)
                edge_idx = edge_idx.cuda(gpu_id)
                edge_weight=edge_weight.cuda(gpu_id)
                model = model.cuda(gpu_id)
        
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        #scheduler = StepLR(optimizer, step_size=10000, gamma=0.1)
        #scheduler = ReduceLROnPlateau(optimizer, 'min')
        
        criterion = torch.nn.MSELoss()
        
        best_val_acc = 0.0
        best_epoch = 0
        #scheduler = ReduceLROnPlateau(optimizer, 'min')
        for epoch in range(0, epoch_n):
            
            model.train()
            optimizer.zero_grad()
            out = model.forward(x, edge_idx, edge_weight) # forwarding x has 0 for single wfs defected ones
            loss = criterion(out[valid_mask_use], lab_out[valid_mask_use]) #for single waveform feature failures (nans etc...)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            #scheduler.step(loss)
            
    
            # if args.eval_enabled: model.eval()
            with torch.no_grad():
                out = model(x, edge_idx, edge_weight)
    
            # Evaluate train
            train_acc = evaluate(out[valid_mask_use], lab_out[valid_mask_use])
            train_mae = evaluate_mae(out[valid_mask_use], lab_out[valid_mask_use])
            # test_acc = evaluate(out[test_mask], labels[test_mask])e
            # val_acc = evaluate(out[val_mask], labels[val_mask])
            if(epoch% 50==0):
                print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, LR : {optimizer.param_groups[0]['lr']:.8f}")
                      
            #early stop criteria
                if(epoch==50):
                    last_val = train_acc.cpu().numpy()
                    flag = 0
                if(epoch>50):
                    curr_val = train_acc.cpu().numpy()
                    val_diff = last_val - curr_val
                    if(abs(val_diff) < 0.0001):
                        flag = flag +1
                    if(flag>2):
                        print('early stop')
                        break
                    last_val = curr_val
    
        train_acc_vec.append(train_acc.cpu().numpy())
        train_mae_vec.append(train_mae.cpu().numpy())
        model_params_vec.append(model.state_dict())
        flag_vec.append(flag)
        
        if(cross_validate):
            with torch.no_grad():
                out = model(x,  edge_idx, edge_weight)
            test_acc = evaluate(out[test_idx], lab_out[test_idx])
            print(f"iteration: {iter_}, test_acc: {test_acc:.4f}")
            test_mae = evaluate_mae(out[test_idx], lab_out[test_idx])
            test_acc_vec.append(test_acc.cpu().numpy())
            test_mae_vec.append(test_mae.cpu().numpy())
        
        
    result = dict()
    result['mse_train']=np.array(train_acc_vec)
    result['mae_train']=np.array(train_mae_vec)
    result['model_params']= model_params_vec
    result['convergence']=np.array(flag_vec)
    
    if(cross_validate):
        result['mse_test']= np.array(test_acc_vec)
        result['mae_test'] = np.array(test_mae_vec)
    return result


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
    
            

def run_gridsearch_batch(nodes, FCs, target_frs, epoch_n, iter_n, model_string, fit_param_list, device):
    
    fit_result=[]
    for entry in fit_param_list:
        fit_params= dict()
        fit_params['dropout_prob']=entry['dropout_prob']
        fit_params['learning_rate']=entry['learning_rate']
        fit_params['weight_decay']=entry['weight_decay']
        
        fit_params['fit_result']=run_GNN_batch_alt(nodes, FCs, target_frs, epoch_n, iter_n, model_string, fit_params, device)
        fit_result.append(fit_params)    
    
    return fit_result

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

def run_gridsearch(features, adj_mat, fr_vec, epoch_n, cuda, iteration, gpu_id, valid_mask, cross_validate, x_ratio, random_seeds, model_string, fit_param_list, transductive):
    
    fit_result=[]
    for entry in fit_param_list:
        fit_params= dict()
        fit_params['dropout_prob']=entry['dropout_prob']
        fit_params['learning_rate']=entry['learning_rate']
        fit_params['weight_decay']=entry['weight_decay']
        
        fit_params['fit_result']=run_GNN_refined(features, adj_mat, fr_vec, epoch_n, cuda, iteration, gpu_id, cross_validate, x_ratio, random_seeds, model_string, fit_params, transductive)
        fit_result.append(fit_params)    
    
    return fit_result

def run_GNN_refined_class(features, adj_mat, fr_vec, epoch_n, cuda, iteration, gpu_id, valid_mask, cross_validate, x_ratio, random_seeds, model_string, fit_params):
    # compute GCN assuming same nodes 
    num_features= features.shape[1]
    
    train_acc_vec=[]
    train_mae_vec=[]
    val_acc_vec=[]
    val_mae_vec=[]
    model_params_vec=[]
    test_acc_vec=[]
    test_mae_vec=[]
    flag_vec=[]
    
    valid_mask_use = np.copy(valid_mask) #backup
    valid_mask_bk = np.copy(valid_mask)
 
    if(len(fr_vec.shape)==1):
         num_classes=1
    else:
         num_classes = fr_vec.shape[1]
    
    for iter_ in range(iteration):
        
        valid_idx = np.arange(len(fr_vec))
        
        np.random.seed(random_seeds[iter_])
        test_idx=np.random.choice(valid_idx, int(features.shape[0]*x_ratio), replace=False)
        train_idx = np.setxor1d(valid_idx, test_idx)
        
        validate_idx =  np.random.choice(train_idx, int(len(train_idx)*x_ratio), replace=False)
        train_idx = np.setxor1d(train_idx, validate_idx)
        #test_idx = np.setxor1d(test_idx, validate_idx)
        
        valid_mask_use = train_idx #getting loss updates only from these
        
        
        
        
        
            
        x = torch.tensor(features)
        lab_out = torch.tensor(fr_vec)
        if(num_classes==1):
            lab_out = torch.reshape(lab_out, (fr_vec.shape[0], 1))
   
        
        edge_idx = np.array(np.where(adj_mat>0))
        edge_idx = torch.tensor(edge_idx)
        edge_weight = adj_mat[np.where(adj_mat>0)]
        edge_weight = torch.tensor(edge_weight)
                    
        
        #model = wp1_gnn_torch.NodeGCN(1,1)
     
        model = return_model(model_string, num_features, num_classes, fit_params['dropout_prob'])
        #model = NodeGCN(num_features,1)
        
        # if(model_string=='rgcn'):
        #     edge_idx = np.array(np.where(adj_mat!=0))
        #     edge_idx = torch.tensor(edge_idx)
        #     edge_type = adj_mat[np.where(adj_mat!=0)]
        #     types = np.unique(edge_type)
        #     edge_class = np.squeeze(np.zeros((edge_type.shape[0],1)))
        #     for jj, typ in enumerate(types):
        #         idx = np.where(edge_type==typ)[0]
        #         edge_class[idx]=jj
        #     edge_weight = torch.tensor(edge_class).type(torch.LongTensor)
            
        
        if(cuda):
            if(gpu_id==0):
                lab_out=lab_out.cuda().long()
                x = x.cuda()
                edge_idx = edge_idx.cuda()
                edge_weight=edge_weight.cuda()
                model = model.cuda()
            else:
                lab_out=lab_out.cuda(gpu_id).long()
                x = x.cuda(gpu_id)
                edge_idx = edge_idx.cuda(gpu_id)
                edge_weight=edge_weight.cuda(gpu_id)
                model = model.cuda(gpu_id)
        
        
        optimizer = torch.optim.Adam(model.parameters(), lr=fit_params['learning_rate'], weight_decay= fit_params['weight_decay'])
        #scheduler = StepLR(optimizer, step_size=10000, gamma=0.1)
        #scheduler = ReduceLROnPlateau(optimizer, 'min')
        if(model_string == 'gcn_class'):
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.MSELoss()
        
        validate_acc_list = []
        #scheduler = ReduceLROnPlateau(optimizer, 'min')
        for epoch in range(0, epoch_n):
            
            model.train()
            optimizer.zero_grad()
            out = model.forward(x, edge_idx, edge_weight) # forwarding x has 0 for single wfs defected ones
            loss = criterion(out[valid_mask_use,:], torch.max(lab_out[valid_mask_use,:],1)[1]) #for single waveform feature failures (nans etc...)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            #scheduler.step(loss)
            
    
            #if args.eval_enabled: 
            #model.eval()
            with torch.no_grad():
                out = model(x, edge_idx, edge_weight)
    
            # Evaluate train
            train_acc = evaluate_acc(out[valid_mask_use,:], lab_out[valid_mask_use,:])
            #train_mae = evaluate_mae(out[valid_mask_use,:], lab_out[valid_mask_use,:])
            validate_acc = evaluate_acc(out[validate_idx,:], lab_out[validate_idx,:])
            #validate_mae = evaluate_mae(out[validate_idx,:], lab_out[validate_idx,:])
            # test_acc = evaluate(out[test_mask], labels[test_mask])e
            # val_acc = evaluate(out[val_mask], labels[val_mask])
            if(epoch% 50==0):
                print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, validate_acc : {validate_acc:.4f}, LR : {optimizer.param_groups[0]['lr']:.8f}")
                      
            #early stop criteria
                # if(epoch==50):
                #     last_val = train_acc.cpu().numpy()
                #     flag = 0
                # if(epoch>50):
                    # curr_val = train_acc.cpu().numpy()
                    # val_diff = last_val - curr_val
                    # if(abs(val_diff) < 0.0001):
                    #     flag = flag +1
                    # if(flag>2):
                    #     print('early stop')
                    #     break
                    # last_val = curr_val
            if(epoch>500):
                flag=0
                validate_acc_list.append(validate_acc.cpu().numpy())
                if(len(validate_acc_list)>10):
                    validate_acc_list.pop(0)
                    validate_mean = np.mean(validate_acc_list)
                    if(validate_acc.cpu().numpy()>validate_mean):
                        print('early stop')
                        flag=1
                        break
                    
    
        train_acc_vec.append(train_acc.cpu().numpy())
        #train_mae_vec.append(train_mae.cpu().numpy())
        val_acc_vec.append(validate_acc.cpu().numpy())
        #val_mae_vec.append(validate_mae.cpu().numpy())
        
        
        model_params_vec.append(model.state_dict())
        flag_vec.append(flag)
        
        with torch.no_grad():
            out = model(x,  edge_idx, edge_weight)
        test_acc = evaluate_acc(out[test_idx,:], lab_out[test_idx,:])
        print(f"iteration: {iter_}, test_acc: {test_acc:.4f}")
        #test_mae = evaluate_mae(out[test_idx,:], lab_out[test_idx,:])
        test_acc_vec.append(test_acc.cpu().numpy())
        #test_mae_vec.append(test_mae.cpu().numpy())
    
        
    result = dict()
    result['acc_train']=np.array(train_acc_vec)
    #result['mae_train']=np.array(train_mae_vec)
    result['model_params']= model_params_vec
    result['convergence']=np.array(flag_vec)
    result['acc_test']= np.array(test_acc_vec)
    #result['mae_test'] = np.array(test_mae_vec)
    result['acc_val']=np.array(val_acc_vec)
    #result['mae_val']=np.array(val_mae_vec)
    return result
    
def standard_scale(features,train_idx, validate_idx, test_idx):
    features_wip = np.copy(features)
    
    if(len(features_wip.shape)==1):
        X_train, X_scaler = neural_assembly_utils.standardscaler_transform(features_wip[train_idx].reshape(-1,1))
        X_validate = X_scaler.transform(features_wip[validate_idx].reshape(-1,1))
        X_test = X_scaler.transform(features_wip[test_idx].reshape(-1,1))
        features_wip[train_idx] = np.squeeze(X_train)
        features_wip[validate_idx] = np.squeeze(X_validate)
        features_wip[test_idx] = np.squeeze(X_test)
    else:    
        X_train, X_scaler = neural_assembly_utils.standardscaler_transform(features_wip[train_idx, :])
        X_validate = X_scaler.transform(features_wip[validate_idx, :])
        X_test = X_scaler.transform(features_wip[test_idx, :])
        features_wip[train_idx, :] = X_train
        features_wip[validate_idx, :] = X_validate
        features_wip[test_idx, :] = X_test
    
    return features_wip
    
    

def run_GNN_refined(features, adj_mat, fr_vec, epoch_n, cuda, iteration, gpu_id, cross_validate, x_ratio, random_seeds, model_string, fit_params, transductive):
    # compute GCN assuming same nodes 
    num_features= features.shape[1]
    
    train_acc_vec=[]
    train_mae_vec=[]
    val_acc_vec=[]
    val_mae_vec=[]
    model_params_vec=[]
    test_acc_vec=[]
    test_mae_vec=[]
    flag_vec=[]
    
    # valid_mask_use = np.copy(valid_mask) #backup
    # valid_mask_bk = np.copy(valid_mask)
    adj_mat_bk = np.copy(adj_mat)
    
    if(len(fr_vec.shape)==1):
         num_classes=1
    else:
         num_classes = fr_vec.shape[1]
    
    for iter_ in range(iteration):
        
        valid_idx = np.arange(len(fr_vec))
        
        np.random.seed(random_seeds[iter_])
        test_idx=np.random.choice(valid_idx, int(features.shape[0]*x_ratio), replace=False)
        train_idx = np.setxor1d(valid_idx, test_idx)
        # train_idx_bk = np.copy(train_idx)
        validate_idx =  np.random.choice(train_idx, int(len(train_idx)*x_ratio), replace=False)
        train_idx = np.setxor1d(train_idx, validate_idx)
        #test_idx = np.setxor1d(test_idx, validate_idx)
        
        valid_mask_use = train_idx #getting loss updates only from these
        
        # trans_tr_idx = np.isin(train_idx_bk, train_idx)
        # trans_val_idx = np.isin(train_idx_bk, validate_idx)
        
        # #standardize
        
        features_wip = standard_scale(features,train_idx, validate_idx, test_idx)
        fr_vec_wip = standard_scale(fr_vec,train_idx, validate_idx, test_idx)
            
        x = torch.tensor(features_wip)
        lab_out = torch.tensor(fr_vec_wip)
        if(num_classes==1):
            lab_out = torch.reshape(lab_out, (fr_vec.shape[0], 1))
        
        if(transductive):
            x_test = torch.tensor(features_wip) # full features for calculating test performance
            lab_out_test = torch.tensor(fr_vec_wip)
            adj_mat_test = np.copy(adj_mat)
            
            x = torch.tensor(features_wip[train_idx, :])
            lab_out = torch.tensor(fr_vec_wip[train_idx])
            adj_mat = adj_mat_bk[train_idx,:]
            adj_mat = adj_mat[:, train_idx]
            
            edge_idx_test = np.array(np.where(adj_mat_test>0))
            edge_idx_test = torch.tensor(edge_idx_test)
            edge_weight_test = adj_mat_test[np.where(adj_mat_test>0)]
            edge_weight_test = torch.tensor(edge_weight_test)
            if(num_classes==1):
                lab_out = torch.reshape(lab_out, (lab_out.shape[0], 1))
                lab_out_test = torch.reshape(lab_out_test, (lab_out_test.shape[0], 1))
            edge_idx_test = edge_idx_test.cuda()
            edge_weight_test = edge_weight_test.cuda()
            lab_out_test = lab_out_test.cuda()
            x_test = x_test.cuda()
        
        edge_idx = np.array(np.where(adj_mat>0))
        edge_idx = torch.tensor(edge_idx)
        edge_weight = adj_mat[np.where(adj_mat>0)]
        edge_weight = torch.tensor(edge_weight)
                    
        
        #model = wp1_gnn_torch.NodeGCN(1,1)
     
        model = return_model(model_string, num_features, num_classes, fit_params['dropout_prob'])
        #model = NodeGCN(num_features,1)
        
        if('rgcn' in model_string):
            edge_idx = np.array(np.where(adj_mat!=0))
            edge_idx = torch.tensor(edge_idx)
            edge_type = adj_mat[np.where(adj_mat!=0)]
            types = np.unique(edge_type)
            edge_class = np.squeeze(np.zeros((edge_type.shape[0],1)))
            for jj, typ in enumerate(types):
                idx = np.where(edge_type==typ)[0]
                edge_class[idx]=jj
            edge_weight = torch.tensor(edge_class).type(torch.LongTensor)
            
        
        if(cuda):
            if(gpu_id==0):
                lab_out=lab_out.cuda()
                x = x.cuda()
                edge_idx = edge_idx.cuda()
                edge_weight=edge_weight.cuda()
                model = model.cuda()
            else:
                lab_out=lab_out.cuda(gpu_id)
                x = x.cuda(gpu_id)
                edge_idx = edge_idx.cuda(gpu_id)
                edge_weight=edge_weight.cuda(gpu_id)
                model = model.cuda(gpu_id)
        
        
        optimizer = torch.optim.Adam(model.parameters(), lr=fit_params['learning_rate'], weight_decay= fit_params['weight_decay'])
        #scheduler = StepLR(optimizer, step_size=10000, gamma=0.1)
        #scheduler = ReduceLROnPlateau(optimizer, 'min')
        if(model_string == 'gcn_class'):
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.MSELoss()
        
        validate_acc_list = []
        #scheduler = ReduceLROnPlateau(optimizer, 'min')
        train_acc_curve=[]
        validate_acc_curve=[]
        for epoch in range(0, epoch_n):
            
            model.train()
            optimizer.zero_grad()
            out = model.forward(x, edge_idx, edge_weight) # forwarding x has 0 for single wfs defected ones
            if(not(transductive)):
                loss = criterion(out[valid_mask_use,:], lab_out[valid_mask_use,:]) #for single waveform feature failures (nans etc...)
            else:
                loss = criterion(out, lab_out)
            
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            #scheduler.step(loss)
            
    
            #if args.eval_enabled: 
            #model.eval()
            with torch.no_grad():
                out = model(x, edge_idx, edge_weight)
    
            # Evaluate train
            
            if(not(transductive)):
                            
                train_acc = evaluate(out[valid_mask_use,:], lab_out[valid_mask_use,:])
                train_mae = evaluate_mae(out[valid_mask_use,:], lab_out[valid_mask_use,:])
                validate_acc = evaluate(out[validate_idx,:], lab_out[validate_idx,:])
                validate_mae = evaluate_mae(out[validate_idx,:], lab_out[validate_idx,:])
            else:
                train_acc = evaluate(out, lab_out)
                train_mae = evaluate_mae(out, lab_out)
                with torch.no_grad():
                    out_test = model(x_test, edge_idx_test, edge_weight_test)
                validate_acc = evaluate(out_test[validate_idx,:], lab_out_test[validate_idx,:])
                validate_mae = evaluate_mae(out_test[validate_idx,:], lab_out_test[validate_idx,:])
                
            # test_acc = evaluate(out[test_mask], labels[test_mask])e
            # val_acc = evaluate(out[val_mask], labels[val_mask])
            if(epoch% 50==0):
                print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, validate_acc : {validate_acc:.4f}, LR : {optimizer.param_groups[0]['lr']:.8f}")
                train_acc_curve.append(train_acc.cpu().numpy())
                validate_acc_curve.append(validate_acc.cpu().numpy())
                      
            #early stop criteria
                # if(epoch==50):
                #     last_val = train_acc.cpu().numpy()
                #     flag = 0
                # if(epoch>50):
                    # curr_val = train_acc.cpu().numpy()
                    # val_diff = last_val - curr_val
                    # if(abs(val_diff) < 0.0001):
                    #     flag = flag +1
                    # if(flag>2):
                    #     print('early stop')
                    #     break
                    # last_val = curr_val
            if(epoch>500):
                flag=0
                validate_acc_list.append(validate_acc.cpu().numpy())
                if(len(validate_acc_list)>10):
                    validate_acc_list.pop(0)
                    validate_mean = np.mean(validate_acc_list)
                    if(validate_acc.cpu().numpy()>validate_mean):
                        print('early stop')
                        flag=1
                        break
                    
    
        train_acc_vec.append(train_acc.cpu().numpy())
        train_mae_vec.append(train_mae.cpu().numpy())
        val_acc_vec.append(validate_acc.cpu().numpy())
        val_mae_vec.append(validate_mae.cpu().numpy())
        
        
        model_params_vec.append(model.state_dict())
        flag_vec.append(flag)
        if(not(transductive)):
            with torch.no_grad():
                out = model(x,  edge_idx, edge_weight)
            test_acc = evaluate(out[test_idx,:], lab_out[test_idx,:])
            print(f"iteration: {iter_}, test_acc: {test_acc:.4f}")
            test_mae = evaluate_mae(out[test_idx,:], lab_out[test_idx,:])
        else:
            with torch.no_grad():
                out_test = model(x_test, edge_idx_test, edge_weight_test)
            test_acc = evaluate(out_test[test_idx,:], lab_out_test[test_idx,:])
            print(f"iteration: {iter_}, test_acc: {test_acc:.4f}")
            test_mae = evaluate_mae(out_test[test_idx,:], lab_out_test[test_idx,:])
            
        
        
        test_acc_vec.append(test_acc.cpu().numpy())
        test_mae_vec.append(test_mae.cpu().numpy())
    
        
    result = dict()
    result['mse_train']=np.array(train_acc_vec)
    result['mae_train']=np.array(train_mae_vec)
    result['model_params']= model_params_vec
    result['convergence']=np.array(flag_vec)
    result['mse_test']= np.array(test_acc_vec)
    result['mae_test'] = np.array(test_mae_vec)
    result['mse_val']=np.array(val_acc_vec)
    result['mae_val']=np.array(val_mae_vec)
    result['train_curve']=np.array(train_acc_curve)
    result['validate_curve']=np.array(validate_acc_curve)
    return result

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
                
def run_GNN_batch_alt(nodes, FCs, target_frs, epoch_n, iter_n, model_string, fit_params, device):
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
        val_acc_vec=[]
        val_mae_vec=[]
        model_params_vec=[]
        test_acc_vec=[]
        test_mae_vec=[]
        flag_vec=[]
        iter_val_idx=[]
        
        # prep x,y 
        target_cp = np.copy(target_frs)
        full_index= np.arange(len(target_frs))
        
        # make x 
        nodes_cp = np.copy(nodes)
        
        
        # FC
        FC_cp = np.copy(FCs)
        
        for iter_ in range(iter_n):
             
            #get validation index 
            train_idx=np.setxor1d(full_index, ii)
            # val_idx = np.random.choice(train_idx, int(len(train_idx)*0.2))[0]
            # train_idx_fin = np.setxor1d(train_idx, val_idx)
            # targets
            test_y = target_cp[ii]
            # val_y = target_cp[val_idx]
            
            train_y = target_cp[train_idx]
            train_y = neural_assembly_utils.flatten_list_1d(train_y)
            
            #features (input)
            train_x, test_x= neural_assembly_utils3.batch_split(nodes_cp, full_index, ii)
            # train_x, val_x = neural_assembly_utils3.batch_split(nodes_cp, train_idx, val_idx)
            
            
            #scale them
            train_x, train_scaler_x=neural_assembly_utils.standardscaler_transform(train_x)
            test_x = train_scaler_x.transform(test_x) 
            
            tr_internal = np.arange(train_x.shape[0])
            val_idx = np.random.choice(tr_internal, int(len(tr_internal)*0.2))
            
            train_idx_fin = np.setxor1d(tr_internal, val_idx)
            
            val_x = train_x[val_idx,:]
            train_x_loss = train_x[train_idx_fin, :]
            
            
            # scale y
            train_y, train_scaler_y=neural_assembly_utils.standardscaler_transform(train_y.reshape(-1,1))
            test_y = train_scaler_y.transform(test_y.reshape(-1,1))
            
            val_y = train_y[val_idx]
            train_y_loss = train_y[train_idx_fin]
            
            # FCs
            train_FC= neural_assembly_utils3.make_diag_batch_FC(FC_cp[train_idx])
            # val_FC = FC_cp[val_idx]
            test_FC = FC_cp[ii]
            # put into cuda 
            train_x = torch.tensor(train_x, device = device)
            train_y = torch.tensor(train_y, device = device)
            val_x = torch.tensor(val_x, device = device)
            val_y = torch.tensor(val_y, device = device)
            test_x = torch.tensor(test_x, device = device)
            test_y = torch.tensor(test_y, device = device)
            
            if(num_classes==1):
                train_y = torch.reshape(train_y, (train_y.shape[0], 1))
                test_y = torch.reshape(test_y, (test_y.shape[0], 1))
                val_y = torch.reshape(val_y, (val_y.shape[0],1))
                
            edge_idx= dict()
            edge_weight =dict()
            edge_idx['train'] = np.array(np.where(train_FC>0))
            edge_idx['train'] = torch.tensor(edge_idx['train'], device = device)
            edge_weight['train'] = train_FC[np.where(train_FC>0)]
            edge_weight['train'] = torch.tensor(edge_weight['train'], device = device)
                        
            #prep for testing 
           
            edge_idx['test'] = np.array(np.where(test_FC>0))
            edge_idx['test'] = torch.tensor(edge_idx['test'], device = device)
            edge_weight['test'] = test_FC[np.where(test_FC>0)]
            edge_weight['test'] = torch.tensor(edge_weight['test'], device = device)
            
            # for validation
            # edge_idx['val'] = np.array(np.where(val_FC>0))
            # edge_idx['val'] = torch.tensor(edge_idx['val'], device = device)
            # edge_weight['val'] = val_FC[np.where(val_FC>0)]
            # edge_weight['val'] = torch.tensor(edge_weight['val'], device = device)
            
            
            model = return_model(model_string, num_features, num_classes, fit_params['dropout_prob'])
            model.to(device)            
            
            if('rgcn' in model_string):
                edge_idx= dict()
                edge_weight =dict()
                
                edge_idx['train'], edge_weight['train'] = make_rgcn_mat(train_FC, device)
                edge_idx['test'], edge_weight['test'] = make_rgcn_mat(test_FC, device)
                # edge_idx['val'], edge_weight['val'] = make_rgcn_mat(val_FC, device)
                
                
            
            
            optimizer = torch.optim.Adam(model.parameters(), lr=fit_params['learning_rate'], weight_decay= fit_params['weight_decay'])
            # optimizer = torch.optim.SGD(model.parameters(), lr=fit_params['learning_rate'], momentum=0, nesterov=False, weight_decay= fit_params['weight_decay'])
            # optimizer = optim.RAdam(model.parameters(), lr=fit_params['learning_rate'], weight_decay= fit_params['weight_decay'])
            
            if(model_string == 'gcn_class'):
                criterion = torch.nn.CrossEntropyLoss()
            else:
                criterion = torch.nn.MSELoss()
                # criterion = torch.nn.SmoothL1Loss()
                        # SmoothL1Loss
            validate_acc_list = []
            train_acc_curve=[]
            validate_acc_curve=[]
            for epoch in range(0, epoch_n):
                
                model.train()
                optimizer.zero_grad()
                out = model.forward(train_x, edge_idx['train'], edge_weight['train']) # forwarding x has 0 for single wfs defected ones
                loss = criterion(out[train_idx_fin], train_y[train_idx_fin])
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                
                #if args.eval_enabled: 
                model.eval()
                
                with torch.no_grad():
                    out=dict()
                    out['train'] = model(train_x, edge_idx['train'], edge_weight['train'])
                    # out['val'] = model(val_x, edge_idx['val'], edge_weight['val'])
                    # out['test'] = model(test_x, edge_idx['test'], edge_weight['test'])
                    
                    # Evaluate train
                    mse=dict()
                    mae=dict()
                    mse['train'] = evaluate(out['train'][train_idx_fin], train_y[train_idx_fin])
                    mae['train'] = evaluate_mae(out['train'][train_idx_fin], train_y[train_idx_fin])
                    
                    mse['val'] = evaluate(out['train'][val_idx], train_y[val_idx])
                    mae['val'] = evaluate_mae(out['train'][val_idx], train_y[val_idx])
                    
                
                
                if(epoch% 50==0):
                    print(f"Epoch: {epoch}, train_acc: {mse['train']:.4f}, validate_acc : {mse['val']:.4f}, LR : {optimizer.param_groups[0]['lr']:.8f}")
                    train_acc_curve.append(mse['train'].cpu().numpy())
                    validate_acc_curve.append(mse['val'].cpu().numpy())
                     # test
                    with torch.no_grad():
                        out['test'] = model(test_x, edge_idx['test'], edge_weight['test'])
                        mse['test'] = evaluate(out['test'], test_y)
                        mae['test'] = evaluate_mae(out['test'], test_y)
                        print(f"iteration: {iter_}, test_acc: {mse['test']:.4f}")
                                  
               
                if(epoch>0):
                    flag=0
                    patience = 0
                    pat_count =0 
                    validate_acc_list.append(mse['val'].cpu().numpy())
                    if(len(validate_acc_list)>10):
                        validate_acc_list.pop(0)
                        validate_mean = np.mean(validate_acc_list)
                        if(mse['val'].cpu().numpy()>validate_mean):
                            pat_count+=1 
                            if(pat_count>patience):
                                print('early stop')
                                flag=1
                                break
                        else:
                            pat_count=0
                        
            # for each iter
            train_acc_vec.append(mse['train'].cpu().numpy())
            train_mae_vec.append(mae['train'].cpu().numpy())
            val_acc_vec.append(mse['val'].cpu().numpy())
            val_mae_vec.append(mae['val'].cpu().numpy())
            iter_val_idx.append(val_idx)
            
            model_dict=dict()
            
            for k,v in model.state_dict().items():
                model_dict[k] =v.cpu()
            
            
            model_params_vec.append(model_dict)
            flag_vec.append(flag)
            
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
        # result['model_params']= model_params_vec
        result['convergence']=np.array(flag_vec)
        result['mse_test']= np.array(test_acc_vec)
        result['mae_test'] = np.array(test_mae_vec)
        result['mse_val']=np.array(val_acc_vec)
        result['mae_val']=np.array(val_mae_vec)
        result['train_curve']=np.array(train_acc_curve)
        result['validate_curve']=np.array(validate_acc_curve)
        result['iter_val_idx'] = np.array(iter_val_idx)
        per_network.append(result)
    return per_network

def run_GNN_batch(nodes, FCs, target_frs, epoch_n, iter_n, model_string, fit_params, device):
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
        val_acc_vec=[]
        val_mae_vec=[]
        model_params_vec=[]
        test_acc_vec=[]
        test_mae_vec=[]
        flag_vec=[]
        iter_val_idx=[]
        
        
        # prep x,y 
        target_cp = np.copy(target_frs)
        full_index= np.arange(len(target_frs))
        
        # make x 
        nodes_cp = np.copy(nodes)
        
        
        # FC
        FC_cp = np.copy(FCs)
        
        for iter_ in range(iter_n):
             
            #get validation index 
            train_idx=np.setxor1d(full_index, ii)
            val_idx = np.random.choice(train_idx, 1)[0]
            train_idx_fin = np.setxor1d(train_idx, val_idx)
            # targets
            test_y = target_cp[ii]
            val_y = target_cp[val_idx]
            train_y = target_cp[train_idx_fin]
            train_y = neural_assembly_utils.flatten_list_1d(train_y)
            
            #features (input)
            train_x, test_x= neural_assembly_utils3.batch_split(nodes_cp, full_index, ii)
            train_x, val_x = neural_assembly_utils3.batch_split(nodes_cp, train_idx, val_idx)
            
            #stack train and val for scaling 
            
            scale_x = np.vstack((train_x, val_x))
            
            
            #scale them
            scaled_x, train_scaler_x=neural_assembly_utils.standardscaler_transform(scale_x)
            test_x = train_scaler_x.transform(test_x) 
            train_x = train_scaler_x.transform(train_x)
            val_x = train_scaler_x.transform(val_x)
            
            # scale y
            scale_y = np.concatenate((train_y, val_y))
            
            scaled_y, train_scaler_y=neural_assembly_utils.standardscaler_transform(scale_y.reshape(-1,1))
            train_y = train_scaler_y.transform(train_y.reshape(-1,1))
            test_y = train_scaler_y.transform(test_y.reshape(-1,1))
            val_y = train_scaler_y.transform(val_y.reshape(-1,1))
            
            # FCs
            train_FC= neural_assembly_utils3.make_diag_batch_FC(FC_cp[train_idx_fin])
            val_FC = FC_cp[val_idx]
            test_FC = FC_cp[ii]
            # put into cuda 
            train_x = torch.tensor(train_x, device = device)
            train_y = torch.tensor(train_y, device = device)
            val_x = torch.tensor(val_x, device = device)
            val_y = torch.tensor(val_y, device = device)
            test_x = torch.tensor(test_x, device = device)
            test_y = torch.tensor(test_y, device = device)
            
            if(num_classes==1):
                train_y = torch.reshape(train_y, (train_y.shape[0], 1))
                test_y = torch.reshape(test_y, (test_y.shape[0], 1))
                val_y = torch.reshape(val_y, (val_y.shape[0],1))
                
            edge_idx= dict()
            edge_weight =dict()
            edge_idx['train'] = np.array(np.where(train_FC>0))
            edge_idx['train'] = torch.tensor(edge_idx['train'], device = device)
            edge_weight['train'] = train_FC[np.where(train_FC>0)]
            edge_weight['train'] = torch.tensor(edge_weight['train'], device = device)
                        
            #prep for testing 
           
            edge_idx['test'] = np.array(np.where(test_FC>0))
            edge_idx['test'] = torch.tensor(edge_idx['test'], device = device)
            edge_weight['test'] = test_FC[np.where(test_FC>0)]
            edge_weight['test'] = torch.tensor(edge_weight['test'], device = device)
            
            # for validation
            edge_idx['val'] = np.array(np.where(val_FC>0))
            edge_idx['val'] = torch.tensor(edge_idx['val'], device = device)
            edge_weight['val'] = val_FC[np.where(val_FC>0)]
            edge_weight['val'] = torch.tensor(edge_weight['val'], device = device)
            
            
            model = return_model(model_string, num_features, num_classes, fit_params['dropout_prob'])
            model.to(device)            
            
            if('rgcn' in model_string):
                edge_idx= dict()
                edge_weight =dict()
                
                edge_idx['train'], edge_weight['train'] = make_rgcn_mat(train_FC, device)
                edge_idx['test'], edge_weight['test'] = make_rgcn_mat(test_FC, device)
                edge_idx['val'], edge_weight['val'] = make_rgcn_mat(val_FC, device)
                
                
            
            
            optimizer = torch.optim.Adam(model.parameters(), lr=fit_params['learning_rate'], weight_decay= fit_params['weight_decay'])
          
            if(model_string == 'gcn_class'):
                criterion = torch.nn.CrossEntropyLoss()
            else:
                criterion = torch.nn.MSELoss()
                # criterion = torch.nn.SmoothL1Loss()
                        # SmoothL1Loss
            validate_acc_list = []
            train_acc_curve=[]
            validate_acc_curve=[]
            for epoch in range(0, epoch_n):
                
                model.train()
                optimizer.zero_grad()
                out = model.forward(train_x, edge_idx['train'], edge_weight['train']) # forwarding x has 0 for single wfs defected ones
                loss = criterion(out, train_y)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                
                #if args.eval_enabled: 
                model.eval()
                
                with torch.no_grad():
                    out=dict()
                    out['train'] = model(train_x, edge_idx['train'], edge_weight['train'])
                    out['val'] = model(val_x, edge_idx['val'], edge_weight['val'])
                    # out['test'] = model(test_x, edge_idx['test'], edge_weight['test'])
                    
                    # Evaluate train
                    mse=dict()
                    mae=dict()
                    mse['train'] = evaluate(out['train'], train_y)
                    mae['train'] = evaluate_mae(out['train'], train_y)
                    
                    mse['val'] = evaluate(out['val'], val_y)
                    mae['val'] = evaluate_mae(out['val'], val_y)
                    
                
                
                if(epoch% 50==0):
                    print(f"Epoch: {epoch}, train_acc: {mse['train']:.4f}, validate_acc : {mse['val']:.4f}, LR : {optimizer.param_groups[0]['lr']:.8f}")
                    train_acc_curve.append(mse['train'].cpu().numpy())
                    validate_acc_curve.append(mse['val'].cpu().numpy())
                          
               
                if(epoch>0):
                    flag=0
                    patience = 0
                    pat_count =0 
                    validate_acc_list.append(mse['val'].cpu().numpy())
                    if(len(validate_acc_list)>10):
                        validate_acc_list.pop(0)
                        validate_mean = np.mean(validate_acc_list)
                        if(mse['val'].cpu().numpy()>validate_mean):
                            pat_count+=1 
                            if(pat_count>patience):
                                print('early stop')
                                flag=1
                                break
                        
            # for each iter
            train_acc_vec.append(mse['train'].cpu().numpy())
            train_mae_vec.append(mae['train'].cpu().numpy())
            val_acc_vec.append(mse['val'].cpu().numpy())
            val_mae_vec.append(mae['val'].cpu().numpy())
            iter_val_idx.append(val_idx)
            
            model_dict=dict()
            
            for k,v in model.state_dict().items():
                model_dict[k] =v.cpu()
            
            
            model_params_vec.append(model_dict)
            flag_vec.append(flag)
            
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
        result['model_params']= model_params_vec
        result['convergence']=np.array(flag_vec)
        result['mse_test']= np.array(test_acc_vec)
        result['mae_test'] = np.array(test_mae_vec)
        result['mse_val']=np.array(val_acc_vec)
        result['mae_val']=np.array(val_mae_vec)
        result['train_curve']=np.array(train_acc_curve)
        result['validate_curve']=np.array(validate_acc_curve)
        result['iter_val_idx'] = np.array(iter_val_idx)
        per_network.append(result)
    return per_network

def run_GNN_batch_x(nodes, FCs, target_frs, n_epoch, iter_n, model_string, fit_params_list, device, chip_ids, gridsearch):
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
        val_acc_vec=[]
        val_mae_vec=[]
        model_params_vec=[]
        test_acc_vec=[]
        test_mae_vec=[]
        flag_vec=[]
        iter_val_idx=[]
        
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
            train_y = neural_assembly_utils.flatten_list_1d(train_y)
            
            # make x 
            #features (input)
            if(gridsearch==0):
                train_x, test_x= batch_split_x(nodes_cp, full_index, ii, same_chip) #identical function to wp1_data_description, wp1_data class
            else:
                train_x, test_x= neural_assembly_utils3.batch_split(nodes_cp, full_index, ii)
                
            #stack train and val for scaling 
            
            #scale them
            scaled_x, train_scaler_x=neural_assembly_utils.standardscaler_transform(train_x)
            test_x = train_scaler_x.transform(test_x) 
            train_x = train_scaler_x.transform(train_x)
            # val_x = train_scaler_x.transform(val_x)
            
            # scale y
            
            scaled_y, train_scaler_y=neural_assembly_utils.standardscaler_transform(train_y.reshape(-1,1))
            train_y = train_scaler_y.transform(train_y.reshape(-1,1))
            test_y = train_scaler_y.transform(test_y.reshape(-1,1))
            # val_y = train_scaler_y.transform(val_y.reshape(-1,1))
            
            # FCs
            train_FC= neural_assembly_utils3.make_diag_batch_FC(FC_cp[train_idx])
            test_FC = FC_cp[ii]
            # put into cuda 
            train_x = torch.tensor(train_x, device = device)
            train_y = torch.tensor(train_y, device = device)
            test_x = torch.tensor(test_x, device = device)
            test_y = torch.tensor(test_y, device = device)
            
            if(num_classes==1):
                train_y = torch.reshape(train_y, (train_y.shape[0], 1))
                test_y = torch.reshape(test_y, (test_y.shape[0], 1))
                
            edge_idx= dict()
            edge_weight =dict()
            edge_idx['train'] = np.array(np.where(train_FC>0))
            edge_idx['train'] = torch.tensor(edge_idx['train'], device = device)
            edge_weight['train'] = train_FC[np.where(train_FC>0)]
            edge_weight['train'] = torch.tensor(edge_weight['train'], device = device)
                        
            #prep for testing 
           
            edge_idx['test'] = np.array(np.where(test_FC>0))
            edge_idx['test'] = torch.tensor(edge_idx['test'], device = device)
            edge_weight['test'] = test_FC[np.where(test_FC>0)]
            edge_weight['test'] = torch.tensor(edge_weight['test'], device = device)
            
            
            
            model = return_model(model_string, num_features, num_classes, fit_params['dropout_prob'], fit_params['hidden_dims'])
            model.to(device)            
            
            if('rgcn' in model_string):
                edge_idx= dict()
                edge_weight =dict()
                edge_idx['train'], edge_weight['train'] = make_rgcn_mat(train_FC, device)
                edge_idx['test'], edge_weight['test'] = make_rgcn_mat(test_FC, device)
                edge_idx['train'] = torch.tensor(edge_idx['train'], device= device)
                edge_idx['test'] = torch.tensor(edge_idx['test'], device= device)
                edge_weight['train'] = torch.tensor(edge_weight['train'], device= device)
                edge_weight['test'] = torch.tensor(edge_weight['test'], device= device)
                
                # edge_idx['val'], edge_weight['val'] = make_rgcn_mat(val_FC, device)
                
            
            optimizer = torch.optim.Adam(model.parameters(), lr=fit_params['learning_rate'], weight_decay= fit_params['weight_decay'])
          
            if(model_string == 'gcn_class'):
                criterion = torch.nn.CrossEntropyLoss()
            else:
                criterion = torch.nn.MSELoss()
                # criterion = torch.nn.SmoothL1Loss()
                        # SmoothL1Loss
            # validate_acc_list = []
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
                
                #if args.eval_enabled: 
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
            # val_acc_vec.append(mse['val'].cpu().numpy())
            # val_mae_vec.append(mae['val'].cpu().numpy())
            # # iter_val_idx.append(val_idx)
            validate_curves_list.append(np.array(validate_acc_curve))
            train_curves_list.append(np.array(train_acc_curve))
            
            model_dict=dict()
            
            for k,v in model.state_dict().items():
                model_dict[k] =v.cpu()
            
            if(gridsearch==0):
                model_params_vec.append(model_dict)
            # flag_vec.append(flag)
            
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
        
        # if(gridsearch==0):
        #     result['model_params']= model_params_vec
        # result['convergence']=np.array(flag_vec)
        result['mse_test']= np.array(test_acc_vec)
        result['mae_test'] = np.array(test_mae_vec)
        # result['mse_val']=np.array(val_acc_vec)
        # result['mae_val']=np.array(val_mae_vec)
        result['train_curve']=train_curves_list
        result['validate_curve']=validate_curves_list
        # result['iter_val_idx'] = np.array(iter_val_idx)
        per_network.append(result)
    return per_network









def run_GNN_samp(features, samp_mats, fr_vec, epoch_n, cuda, iteration, gpu_id, valid_mask, cross_validate, x_ratio, random_seeds, model_string):
    # compute GCN assuming same nodes 
    num_features= features.shape[1]
    

    train_acc_vec=[]
    train_mae_vec=[]
    model_params_vec=[]
    
    flag_vec=[]
    
    valid_mask_use = np.copy(valid_mask) #backup
    valid_mask_bk = np.copy(valid_mask)
    
    samp_test_idx = []
    
    model = return_model(model_string, num_features)
    if(cuda):
        if(gpu_id==0):
            model = model.cuda()
        else:
            model = model.cuda(gpu_id)
            
    
    #scheduler = StepLR(optimizer, step_size=10000, gamma=0.1)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    
    
    
    
    for iter_, samp_mat in enumerate(samp_mats):
        
        adj_mat = samp_mat['masked_eff']
        #scheduler = ReduceLROnPlateau(optimizer, 'min')
    
        
        if(cross_validate):
            valid_idx = np.where(valid_mask==True)[0]
            np.random.seed(random_seeds[iter_])
            test_idx=np.random.choice(valid_idx, int(features.shape[0]*x_ratio), replace=False)
            train_idx = np.setxor1d(valid_idx, test_idx)
            
            valid_mask_use = train_idx #getting loss updates only from these
            samp_test_idx.append(test_idx)
        else:
            valid_mask_use = valid_mask_bk
            
        x = torch.tensor(features)
        lab_out = torch.tensor(fr_vec)
        lab_out = torch.reshape(lab_out, (features.shape[0], 1))
       
        
        edge_idx = np.array(np.where(adj_mat>0))
        edge_idx = torch.tensor(edge_idx)
        edge_weight = adj_mat[np.where(adj_mat>0)]
        edge_weight = torch.tensor(edge_weight)
        
        if(cuda):
            if(gpu_id==0):
                lab_out=lab_out.cuda()
                x = x.cuda()
                edge_idx = edge_idx.cuda()
                edge_weight=edge_weight.cuda()
                
            else:
                lab_out=lab_out.cuda(gpu_id)
                x = x.cuda(gpu_id)
                edge_idx = edge_idx.cuda(gpu_id)
                edge_weight=edge_weight.cuda(gpu_id)
        
        for epoch in range(0, epoch_n):
            
            model.train()
            optimizer.zero_grad()
            out = model.forward(x, edge_idx, edge_weight) # forwarding x has 0 for single wfs defected ones
            loss = criterion(out[valid_mask_use], lab_out[valid_mask_use]) #for single waveform feature failures (nans etc...)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            #scheduler.step(loss)
            
    
            # if args.eval_enabled: model.eval()
            with torch.no_grad():
                out = model(x, edge_idx, edge_weight)
    
            # Evaluate train
            train_acc = evaluate(out[valid_mask_use], lab_out[valid_mask_use])
            train_mae = evaluate_mae(out[valid_mask_use], lab_out[valid_mask_use])
            # test_acc = evaluate(out[test_mask], labels[test_mask])
            # val_acc = evaluate(out[val_mask], labels[val_mask])
            if(epoch% 50==0):
                print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, LR : {optimizer.param_groups[0]['lr']:.8f}")
                      
            #early stop criteria
                if(epoch==50):
                    last_val = train_acc.cpu().numpy()
                    flag = 0
                if(epoch>50):
                    curr_val = train_acc.cpu().numpy()
                    val_diff = last_val - curr_val
                    if(abs(val_diff) < 0.0001):
                        flag = flag +1
                    if(flag>2):
                        print('early stop')
                        break
                    last_val = curr_val
    
        train_acc_vec.append(train_acc.cpu().numpy())
        train_mae_vec.append(train_mae.cpu().numpy())
        model_params_vec.append(model.state_dict())
        flag_vec.append(flag)
        
    if(cross_validate):
        test_acc_vec=[]
        test_mae_vec=[]
        
        
        for iter_, samp_mat in enumerate(samp_mats):
            adj_mat = samp_mat['masked_eff']
            x = torch.tensor(features)
            lab_out = torch.tensor(fr_vec)
            lab_out = torch.reshape(lab_out, (features.shape[0], 1))
           
            
            edge_idx = np.array(np.where(adj_mat>0))
            edge_idx = torch.tensor(edge_idx)
            edge_weight = adj_mat[np.where(adj_mat>0)]
            edge_weight = torch.tensor(edge_weight)
                        
            
            #model = wp1_gnn_torch.NodeGCN(1,1)
           
            #model = NodeGCN(num_features,1)
                
            
            if(cuda):
                if(gpu_id==0):
                    lab_out=lab_out.cuda()
                    x = x.cuda()
                    edge_idx = edge_idx.cuda()
                    edge_weight=edge_weight.cuda()
                    
                else:
                    lab_out=lab_out.cuda(gpu_id)
                    x = x.cuda(gpu_id)
                    edge_idx = edge_idx.cuda(gpu_id)
                    edge_weight=edge_weight.cuda(gpu_id)
                    
        
            test_idx = samp_test_idx[iter_]
        
            with torch.no_grad():
                out = model(x,  edge_idx, edge_weight)
            test_acc = evaluate(out[test_idx], lab_out[test_idx])
            print(f"iteration: {iter_}, test_acc: {test_acc:.4f}")
            test_mae = evaluate_mae(out[test_idx], lab_out[test_idx])
            test_acc_vec.append(test_acc.cpu().numpy())
            test_mae_vec.append(test_mae.cpu().numpy())
        
        
    result = dict()
    result['mse_train']=np.array(train_acc_vec)
    result['mae_train']=np.array(train_mae_vec)
    result['model_params']= model_params_vec
    result['convergence']=np.array(flag_vec)
    
    if(cross_validate):
        result['mse_test']= np.array(test_acc_vec)
        result['mae_test'] = np.array(test_mae_vec)
    return result

def run_GNN_sampnet(features, samp_mats, samp_target, epoch_n, cuda, iteration, gpu_id, valid_mask, cross_validate, x_ratio, random_seeds, model_string):
    # compute GCN assuming same nodes 
    num_features= features.shape[1]
    

    train_acc_vec=[]
    train_mae_vec=[]
    model_params_vec=[]
    
    flag_vec=[]
    
    valid_mask_use = np.copy(valid_mask) #backup
    valid_mask_bk = np.copy(valid_mask)
    
       
    n_loo = len(samp_mats) #loo regime 
    test_acc_vec=[]
    test_mae_vec=[]
    for n_ in range(n_loo):
        
        model = return_model(model_string, num_features)
        if(cuda):
            if(gpu_id==0):
                model = model.cuda()
            else:
                model = model.cuda(gpu_id)
                
        
        #scheduler = StepLR(optimizer, step_size=10000, gamma=0.1)
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
       
        samp_test_idx = n_
        tr_samp_mats = np.delete(np.array(samp_mats), samp_test_idx)
        tr_samp_target = np.delete(np.asarray(samp_target), samp_test_idx, axis=0) 
        #maybe here shuffling...
        shuf_idx = np.random.permutation(np.arange(len(tr_samp_mats))) 
        tr_samp_mats = tr_samp_mats[shuf_idx]
        tr_samp_target = tr_samp_target[shuf_idx, :]
            
        for ii, samp_mat in enumerate(tr_samp_mats):
            
            adj_mat = samp_mat['masked_eff']
            #scheduler = ReduceLROnPlateau(optimizer, 'min')
            fr_vec = np.squeeze(tr_samp_target[ii,:])
            
            valid_mask_use = valid_mask_bk
                
            x = torch.tensor(features)
            lab_out = torch.tensor(fr_vec)
            lab_out = torch.reshape(lab_out, (features.shape[0], 1))
           
            
            edge_idx = np.array(np.where(adj_mat>0))
            edge_idx = torch.tensor(edge_idx)
            edge_weight = adj_mat[np.where(adj_mat>0)]
            edge_weight = torch.tensor(edge_weight)
            
            if(cuda):
                if(gpu_id==0):
                    lab_out=lab_out.cuda()
                    x = x.cuda()
                    edge_idx = edge_idx.cuda()
                    edge_weight=edge_weight.cuda()
                    
                else:
                    lab_out=lab_out.cuda(gpu_id)
                    x = x.cuda(gpu_id)
                    edge_idx = edge_idx.cuda(gpu_id)
                    edge_weight=edge_weight.cuda(gpu_id)
            
            for epoch in range(0, epoch_n):
                
                model.train()
                optimizer.zero_grad()
                out = model.forward(x, edge_idx, edge_weight) # forwarding x has 0 for single wfs defected ones
                loss = criterion(out[valid_mask_use], lab_out[valid_mask_use]) #for single waveform feature failures (nans etc...)
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                #scheduler.step(loss)
                
        
                # if args.eval_enabled: model.eval()
                with torch.no_grad():
                    out = model(x, edge_idx, edge_weight)
        
                # Evaluate train
                train_acc = evaluate(out[valid_mask_use], lab_out[valid_mask_use])
                train_mae = evaluate_mae(out[valid_mask_use], lab_out[valid_mask_use])
                # test_acc = evaluate(out[test_mask], labels[test_mask])
                # val_acc = evaluate(out[val_mask], labels[val_mask])
                if(epoch% 50==0):
                    print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, LR : {optimizer.param_groups[0]['lr']:.8f}")
                          
                #early stop criteria
                    if(epoch==50):
                        last_val = train_acc.cpu().numpy()
                        flag = 0
                    if(epoch>50):
                        curr_val = train_acc.cpu().numpy()
                        val_diff = last_val - curr_val
                        if(abs(val_diff) < 0.01):
                            flag = flag +1
                        if(flag>2):
                            print('early stop')
                            break
                        last_val = curr_val
        
            train_acc_vec.append(train_acc.cpu().numpy())
            train_mae_vec.append(train_mae.cpu().numpy())
            model_params_vec.append(model.state_dict())
            flag_vec.append(flag)
        
        
        
        
       
        adj_mat = samp_mats[samp_test_idx]['masked_eff']
        fr_vec = samp_target[samp_test_idx]
        x = torch.tensor(features)
        lab_out = torch.tensor(fr_vec)
        lab_out = torch.reshape(lab_out, (features.shape[0], 1))
       
        
        edge_idx = np.array(np.where(adj_mat>0))
        edge_idx = torch.tensor(edge_idx)
        edge_weight = adj_mat[np.where(adj_mat>0)]
        edge_weight = torch.tensor(edge_weight)
                    
              
        
        if(cuda):
            if(gpu_id==0):
                lab_out=lab_out.cuda()
                x = x.cuda()
                edge_idx = edge_idx.cuda()
                edge_weight=edge_weight.cuda()
                
            else:
                lab_out=lab_out.cuda(gpu_id)
                x = x.cuda(gpu_id)
                edge_idx = edge_idx.cuda(gpu_id)
                edge_weight=edge_weight.cuda(gpu_id)
                
    
        with torch.no_grad():
            out = model(x,  edge_idx, edge_weight)
        test_acc = evaluate(out[valid_mask_use], lab_out[valid_mask_use])
        print(f"iteration: {n_}, test_acc: {test_acc:.4f}")
        test_mae = evaluate_mae(out[valid_mask_use], lab_out[valid_mask_use])
        test_acc_vec.append(test_acc.cpu().numpy())
        test_mae_vec.append(test_mae.cpu().numpy())
    
        
    result = dict()
    result['mse_train']=np.array(train_acc_vec)
    result['mae_train']=np.array(train_mae_vec)
    result['model_params']= model_params_vec
    result['convergence']=np.array(flag_vec)
    
   
    result['mse_test']= np.array(test_acc_vec)
    result['mae_test'] = np.array(test_mae_vec)
    return result

def prepare_batch_data(features, tr_samp_mats, tr_samp_target, valid_mask):
    concat_n = len(tr_samp_mats)
    index_shift = features.shape[0]
    x_input = np.tile(features, (concat_n,1))
    y_output = np.concatenate(tr_samp_target, axis=0)
    mask = np.tile(valid_mask, concat_n)
    global_index=[]
    global_weight=[]
    for ii, entry in enumerate(tr_samp_mats):
        
        mat = entry['masked_eff']
        edge_idx = np.array(np.where(mat>0))
        global_index.append(edge_idx + index_shift*ii)
        global_weight.append(mat[np.where(mat>0)])
        
    global_index = np.concatenate(global_index, axis=1)
    global_weight= np.concatenate(global_weight, axis=0)
    
    batch_data =dict()
    batch_data['input']=x_input
    batch_data['target']=y_output
    batch_data['edges']=global_index
    batch_data['weights']=global_weight
    batch_data['mask'] = mask
    
    return batch_data
    

def run_GNN_samp_batch(features, samp_mats, samp_target, epoch_n, cuda, iteration, gpu_id, valid_mask, cross_validate, x_ratio, random_seeds, model_string):
    # compute GCN assuming same nodes 
    num_features= features.shape[1]
    

    train_acc_vec=[]
    train_mae_vec=[]
    model_params_vec=[]
    
    flag_vec=[]
    
    valid_mask_use = np.copy(valid_mask) #backup
    valid_mask_bk = np.copy(valid_mask)
    
       
    n_loo = len(samp_mats) #loo regime 
    test_acc_vec=[]
    test_mae_vec=[]
    for n_ in range(n_loo):
        
        model = return_model(model_string, num_features)
        if(cuda):
            if(gpu_id==0):
                model = model.cuda()
            else:
                model = model.cuda(gpu_id)
                
        
        #scheduler = StepLR(optimizer, step_size=10000, gamma=0.1)
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        #scheduler = ReduceLROnPlateau(optimizer, 'min')
        samp_test_idx = n_
        tr_samp_mats = np.delete(np.array(samp_mats), samp_test_idx)
        tr_samp_target = np.delete(np.asarray(samp_target), samp_test_idx, axis=0) 
        #maybe here shuffling...
        shuf_idx = np.random.permutation(np.arange(len(tr_samp_mats))) 
        tr_samp_mats = tr_samp_mats[shuf_idx]
        tr_samp_target = tr_samp_target[shuf_idx, :]
        
        # generate batch
        batch_data = prepare_batch_data(features, tr_samp_mats, tr_samp_target, valid_mask)
            
        #beginning of batch training 
        
        valid_mask_use = batch_data['mask']
            
        x = torch.tensor(batch_data['input'])
        lab_out = torch.tensor(batch_data['target'])
        lab_out = torch.reshape(lab_out, (batch_data['target'].shape[0], 1))
        edge_idx = torch.tensor(batch_data['edges'])
        edge_weight = torch.tensor(batch_data['weights'])
        
        if(cuda):
            if(gpu_id==0):
                lab_out=lab_out.cuda()
                x = x.cuda()
                edge_idx = edge_idx.cuda()
                edge_weight=edge_weight.cuda()
                
            else:
                lab_out=lab_out.cuda(gpu_id)
                x = x.cuda(gpu_id)
                edge_idx = edge_idx.cuda(gpu_id)
                edge_weight=edge_weight.cuda(gpu_id)
        
        for epoch in range(0, epoch_n):
            
            model.train()
            optimizer.zero_grad()
            out = model.forward(x, edge_idx, edge_weight) # forwarding 
            loss = criterion(out[valid_mask_use], lab_out[valid_mask_use]) #for single waveform feature failures (nans etc...)
            # loss = criterion(out[valid_mask], lab_out[valid_mask]) #for single waveform feature failures (nans etc...)
            
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            #scheduler.step(loss)
            
    
            # if args.eval_enabled: model.eval()
            with torch.no_grad():
                out = model(x, edge_idx, edge_weight)
    
            # Evaluate train
            train_acc = evaluate(out[valid_mask_use], lab_out[valid_mask_use])
            train_mae = evaluate_mae(out[valid_mask_use], lab_out[valid_mask_use])
            
            # train_acc = evaluate(out[valid_mask], lab_out[valid_mask])
            # train_mae = evaluate_mae(out[valid_mask], lab_out[valid_mask])
            
            
            
            # test_acc = evaluate(out[test_mask], labels[test_mask])
            # val_acc = evaluate(out[val_mask], labels[val_mask])
            if(epoch% 50==0):
                print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, LR : {optimizer.param_groups[0]['lr']:.8f}")
                      
            #early stop criteria
                if(epoch==50):
                    last_val = train_acc.cpu().numpy()
                    flag = 0
                if(epoch>50):
                    curr_val = train_acc.cpu().numpy()
                    val_diff = last_val - curr_val
                    if(abs(val_diff) < 0.00001):
                        flag = flag +1
                    if(flag>2):
                        print('early stop')
                        break
                    last_val = curr_val
    
        train_acc_vec.append(train_acc.cpu().numpy())
        train_mae_vec.append(train_mae.cpu().numpy())
        model_params_vec.append(model.state_dict())
        flag_vec.append(flag)
    
        
        
        
       
        adj_mat = samp_mats[samp_test_idx]['masked_eff']
        fr_vec = samp_target[samp_test_idx]
        x = torch.tensor(features)
        lab_out = torch.tensor(fr_vec)
        lab_out = torch.reshape(lab_out, (features.shape[0], 1))
       
        
        edge_idx = np.array(np.where(adj_mat>0))
        edge_idx = torch.tensor(edge_idx)
        edge_weight = adj_mat[np.where(adj_mat>0)]
        edge_weight = torch.tensor(edge_weight)
                    
              
        
        if(cuda):
            if(gpu_id==0):
                lab_out=lab_out.cuda()
                x = x.cuda()
                edge_idx = edge_idx.cuda()
                edge_weight=edge_weight.cuda()
                
            else:
                lab_out=lab_out.cuda(gpu_id)
                x = x.cuda(gpu_id)
                edge_idx = edge_idx.cuda(gpu_id)
                edge_weight=edge_weight.cuda(gpu_id)
                
    
        with torch.no_grad():
            out = model(x,  edge_idx, edge_weight)
        test_acc = evaluate(out[valid_mask], lab_out[valid_mask])
        print(f"iteration: {n_}, test_acc: {test_acc:.4f}")
        test_mae = evaluate_mae(out[valid_mask], lab_out[valid_mask])
        test_acc_vec.append(test_acc.cpu().numpy())
        test_mae_vec.append(test_mae.cpu().numpy())
    
        
    result = dict()
    result['mse_train']=np.array(train_acc_vec)
    result['mae_train']=np.array(train_mae_vec)
    result['model_params']= model_params_vec
    result['convergence']=np.array(flag_vec)
    
   
    result['mse_test']= np.array(test_acc_vec)
    result['mae_test'] = np.array(test_mae_vec)
    return result






def run_Mlpsimple(features, fr_vec, epoch_n, cuda, iteration, gpu_id, cross_validate, slp, x_ratio, random_seeds, valid_mask):
    
    
    train_acc_vec=[]
    model_params_vec=[]
    
    test_acc_vec=[]
    flag_vec=[]
    
    
    num_features= features.shape[1]
            
    
    for iter_ in range(iteration):
        if(cross_validate):
            
            valid_idx = np.where(valid_mask==True)[0]
            np.random.seed(random_seeds[iter_])
            test_idx=np.random.choice(valid_idx, int(features.shape[0]*x_ratio), replace=False)
            train_idx = np.setxor1d(valid_idx, test_idx)
            
            X_train = features[train_idx, :]
            X_test = features[test_idx, :]
            y_train = fr_vec[train_idx]
            y_test= fr_vec[test_idx]
            
            
           # X_train, X_test, y_train, y_test=train_test_split(features, fr_vec, test_size=x_ratio, random_state = random_seeds[iter_])
            x = torch.tensor(X_train)
            lab_out = torch.tensor(y_train)
            lab_out = torch.reshape(lab_out, (lab_out.shape[0], 1))
            x_test = torch.tensor(X_test)
            lab_test = torch.tensor(y_test)
            lab_test = torch.reshape(lab_test, (lab_test.shape[0],1))
        
        else:
            x = torch.tensor(features[valid_mask, :])
            lab_out = torch.tensor(fr_vec[valid_mask])
            lab_out = torch.reshape(lab_out, (lab_out.shape[0], 1))
        
        #model = wp1_gnn_torch.NodeGCN(1,1)
        if(slp):
            model = Slpsimple(num_features,1)
        else:
            model = Mlpsimple(num_features,1)
        
        
        
        model= model.double()
        if(cuda):
            lab_out=lab_out.cuda(gpu_id)
            x = x.cuda(gpu_id)
            model = model.cuda(gpu_id)
            x_test = torch.tensor(X_test).cuda(gpu_id)
            lab_test = torch.tensor(y_test).cuda(gpu_id)
        
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        #scheduler = StepLR(optimizer, step_size=10000, gamma=0.1)
        #scheduler = ReduceLROnPlateau(optimizer, 'min')
        
        criterion = torch.nn.MSELoss()
        
        best_val_acc = 0.0
        best_epoch = 0
        scheduler = ReduceLROnPlateau(optimizer, 'min')
        for epoch in range(0, epoch_n):
            
            model.train()
            optimizer.zero_grad()
            out = model.forward(x) # forwarding x has 0 for single wfs defected ones
            loss = criterion(out, lab_out) #for single waveform feature failures (nans etc...)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step(loss)
            
    
            # if args.eval_enabled: model.eval()
            with torch.no_grad():
                out = model(x)
    
            # Evaluate train
            train_acc = evaluate(out, lab_out)
            # test_acc = evaluate(out[test_mask], labels[test_mask])e
            # val_acc = evaluate(out[val_mask], labels[val_mask])
            if(epoch% 50==0):
                print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, LR : {optimizer.param_groups[0]['lr']:.8f}")
                      
            #early stop criteria
                if(epoch==50):
                    last_val = train_acc.cpu().numpy()
                    flag = 0
                if(epoch>50):
                    curr_val = train_acc.cpu().numpy()
                    val_diff = last_val - curr_val
                    if(abs(val_diff) < 0.0001):
                        flag = flag +1
                    if(flag>2):
                        print('early stop')
                        break
                    last_val = curr_val
    
        train_acc_vec.append(train_acc.cpu().numpy())
        model_params_vec.append(model.state_dict())
        flag_vec.append(flag)
        if(cross_validate):
            
            with torch.no_grad():
                out_test = model(x_test)
            
            test_acc = evaluate(out_test, lab_test)
            test_acc_vec.append(test_acc.cpu().numpy())
            print(f"iteration: {iter_}, test_acc: {test_acc:.4f}")
            
        
    
    result = dict()
    result['mse_loss']=np.array(train_acc_vec)
    result['model_params']= model_params_vec
    result['convergence'] = np.array(flag_vec)
    
    
    if(cross_validate):
        result['test_loss']= np.array(test_acc_vec)
    

        
    return result







def run_gnnexplainer(model, epochs, input_vec, node_idx, adj_mat, target_vec, cuda, gpu_id):
    
    x = torch.tensor(input_vec)
    lab_out = torch.tensor(target_vec)
    lab_out = torch.reshape(lab_out, (adj_mat.shape[0], 1))
    edge_idx = np.array(np.where(adj_mat>0))
    edge_idx = torch.tensor(edge_idx)
    edge_weight = adj_mat[np.where(adj_mat>0)]
    edge_weight = torch.tensor(edge_weight)
    
    
    
    if(cuda):
        lab_out=lab_out.cuda(gpu_id)
        x = x.cuda(gpu_id)
        edge_idx = edge_idx.cuda(gpu_id)
        edge_weight=edge_weight.cuda(gpu_id)
        model = model.cuda(gpu_id).double()
    else:
        model = model.cpu() # check for double vs float concerns 
        
   
    
    explainer = GNNExplainer(model, epochs=epochs, num_hops=3)
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_idx, edge_weights=edge_weight)
    # ax, G = explainer.visualize_subgraph(node_idx, edge_idx, edge_mask, y=lab_out, threshold=0.9)
    # plt.show()
    
    return node_feat_mask.cpu().detach().numpy(), edge_mask.cpu().detach().numpy()




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



#PGexplainer
# explainer = PGExplainer(model, edge_idx, x, 'node', edge_weight)
# node_idx = [5,10]
# explainer.prepare(node_idx)
# graph, expl = explainer.explain(5)

#from ExplanationEvaluation.utils.plotting import plot
#plot(graph, expl, fr_vec, 10, 12, 300, 'else', show=True) #total number of edges : thresh min (12), numper of top edges: 100 


#GNN explainer
# explainer = GNNExplainer(model, epochs=200)
# node_idx = 10
# node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_idx, edge_weights=edge_weight)
# ax, G = explainer.visualize_subgraph(node_idx, edge_idx, edge_mask, y=lab_out)
# plt.show()
        


def get_gcn_train_test_idx(features, adj_mat, fr_vec, valid_mask, x_ratio, random_seeds, iter_):
    # compute GCN assuming same nodes 
    

    valid_mask_use = np.copy(valid_mask) #backup
    valid_mask_bk = np.copy(valid_mask)
    
    
    valid_idx = np.where(valid_mask==True)[0]
    np.random.seed(random_seeds[iter_])
    test_idx=np.random.choice(valid_idx, int(features.shape[0]*x_ratio), replace=False)
    train_idx = np.delete(np.arange(len(valid_mask)), test_idx)
    
   

    return train_idx, test_idx