#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:23:59 2021

@author: th
"""


import torch
from torch.nn import ReLU, Linear, Softmax, SmoothL1Loss, Tanh, LeakyReLU
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool, SGConv, SAGEConv, GATConv, FastRGCNConv, GraphConv
import torch.nn.functional as F



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
       out1 = F.dropout(out1, training=self.training, p=self.dropout_prob)
       out1 = self.relu1(out1)
       
       stack.append(out1)

       out2 = self.conv2(out1, edge_index, edge_type)
       out2= F.dropout(out2,training=self.training, p=self.dropout_prob)
       out2 = self.relu2(out2)
       stack.append(out2)

       out3 = self.conv3(out2, edge_index, edge_type)
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
        out1 = F.dropout(out1, training=self.training, p=self.dropout_prob)
        out1 = self.relu1(out1)
        stack.append(out1)
        
        out2 = self.conv2(out1, edge_index, edge_type)
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
        out1 = F.dropout(out1, training=self.training, p=self.dropout_prob)
        out1 = self.relu1(out1)
        stack.append(out1)
        # out1 = self.tanh1(out1)
        
        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = F.dropout(out2, training=self.training, p=self.dropout_prob)
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
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
        out1 = F.dropout(out1, training=self.training, p=self.dropout_prob)
        out1 = self.relu1(out1)
        # out1 = self.tanh1(out1)
        stack.append(out1)
        
        out2 = self.conv2(out1, edge_index, edge_weights)
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
        out1 = F.dropout(out1, training=self.training, p=self.dropout_prob)
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = F.dropout(out2, training=self.training, p=self.dropout_prob)
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
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
        out1 = F.dropout(out1, training=self.training, p=self.dropout_prob)
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
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
        out1 = F.dropout(out1, training=self.training, p=self.dropout_prob)
        out1 = self.relu1(out1)
        stack.append(out1)

        input_lin = torch.cat(stack, dim=1)

        return input_lin





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
    

