import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv
from torch_sparse import coalesce, SparseTensor
import torch.optim as optim
from torch_geometric.nn import global_add_pool,global_mean_pool,AttentionalAggregation
from torch.autograd import grad
from itertools import chain
from torchviz import make_dot
import sys
from tqdm import tqdm

import os.path as osp
import GCL.losses as L
import GCL.augmentors as A

from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader,Data

from torchmetrics import AUROC,Accuracy
import numpy as np
import pandas as pd
import random
import string
from termcolor import colored
from copy import deepcopy

from modelNew.base_model import BaseModel
from modelNew.ssl_module import Encoder
from modelNew.gcn import GCN
from modelNew.gin import GIN
from modelNew.vgin import vGIN
from modelNew.utils import *


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import torch
import torchmetrics


import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data



class Model(BaseModel):
    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5,edge_dim = -1,save_mem=True,jk='last',node_cls=False,pooling='sum',
                with_bn=False, weight_decay=5e-6,lr=1e-3,adapt_lr=1e-4,lr_scheduler=True,patience=50,early_stop_epochs=5,lr_decay=0.75,penalty=0.1,gradMatching_penalty=1.0,project_layer_num=2,edge_gnn='none',edge_gnn_layers=2,edge_budget=0.75,num_samples=1,edge_penalty=1.0,edge_uniform_penalty=1e-2,edge_prob_thres=50,featureMasking=True,temp=0.2,adapt_params='edge_gnn', with_bias=True,base_gnn='gin',valid_metric='acc', device='cpu', **args):

        super(Model, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.rnd_id = ''.join(random.choices(string.digits, k=16))  # for caching stuffs
        self.debug = args['debug']
        self.useAutoAug = args['useAutoAug']
        dataset_name = args.get('dataset_name')
        use_mol_encoder = args.get('mol_encoder',False)
        use_vGIN = args.get('use_vGIN',False)
        self.use_rsc = args.get('use_rsc',False)
        self.p_rsc = args.get('p_rsc',0.3)
        self.use_div_cls = args.get('use_div_cls',False)
        self.div_headers = args.get('div_headers',10)
        self.div_reg = args.get('div_reg',1.0)
        self.uniform_p = args.get('uniform_p',-1.0)
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_samples = num_samples
        self.best_states = None
        self.best_meta_model = None
        self.device = device
        self.jk = jk
        self.node_cls = node_cls
        self.edge_dim = edge_dim
        self.cls_header = self.create_mlp(nhid,nhid,nclass,project_layer_num)
        self.ssl_header = self.create_mlp(nhid,nhid,nclass,project_layer_num,cls=False)
        self.linear_refit_layer = self.create_mlp(nhid,nhid,nclass,project_layer_num)
        self.meta_linear_cls = nn.Linear(nhid, nclass)

        self.penalty = penalty
        self.gradMatching_penalty = gradMatching_penalty
        self.edge_budget = edge_budget
        self.edge_penalty = edge_penalty
        self.edge_uniform_penalty = edge_uniform_penalty
        self.featureMasking = featureMasking
        self.edge_prob_thres = edge_prob_thres

        self.featsMask = nn.Parameter(torch.zeros(nfeat)+5.0).view(1,-1).to(device)
        self.label_emb = nn.Embedding(nclass, nhid)
        
        #  pretraining epochs
        self.pe = args['pretraining_epochs']
        
        print (colored(f'Use vGIN:{use_vGIN}','blue'))
        
        
        if base_gnn=='gin':
            if use_vGIN:
                self.gnn = vGIN(nfeat, nhid, nclass, nlayers, dropout=dropout,edge_dim=edge_dim,jk=jk,node_cls=node_cls,pooling=pooling,
                with_bn=with_bn, weight_decay=weight_decay,dataset_name=dataset_name,mol_encoder = use_mol_encoder)
            else:
                self.gnn = GIN(nfeat, nhid, nclass, nlayers, dropout=dropout,edge_dim=edge_dim,jk=jk,node_cls=node_cls,pooling=pooling,
                    with_bn=with_bn, weight_decay=weight_decay,dataset_name=dataset_name,mol_encoder = use_mol_encoder)
        if base_gnn=='gcn':
            self.gnn = GCN(nfeat, nhid,nclass, nlayers=nlayers, edge_dim=edge_dim,dropout=dropout,save_mem=save_mem,jk=jk,node_cls=node_cls,pooling=pooling,
                with_bn=with_bn, weight_decay=weight_decay, with_bias=with_bias,dataset_name=dataset_name, mol_encoder = use_mol_encoder)

        self.edge_gnn = edge_gnn
        if edge_gnn=='gin':
            if use_vGIN:
                self.dataAug_gnn = vGIN(nfeat, nhid, nclass, nlayers=edge_gnn_layers, edge_dim=edge_dim,dropout=dropout,jk='last',node_cls=True,
                    with_bn=with_bn, weight_decay=weight_decay,dataset_name=dataset_name, mol_encoder = use_mol_encoder)
            else:
                self.dataAug_gnn = GIN(nfeat, nhid, nclass, nlayers=edge_gnn_layers, edge_dim=edge_dim,dropout=dropout,jk='last',node_cls=True,
                    with_bn=with_bn, weight_decay=weight_decay,dataset_name=dataset_name, mol_encoder = use_mol_encoder)
        if edge_gnn=='gcn':
            self.dataAug_gnn = GCN(nfeat, nhid,nclass, edge_dim=edge_dim, nlayers=edge_gnn_layers, dropout=dropout,save_mem=save_mem,jk='last',node_cls=True,
                with_bn=with_bn, weight_decay=weight_decay, with_bias=with_bias,dataset_name=dataset_name, mol_encoder = use_mol_encoder)
        if edge_gnn != 'none':
            self.edge_linear = nn.Sequential(nn.Linear(3*nhid, 2*nhid),nn.ReLU(),nn.Linear(2*nhid, 2))
        
        self.edge_loss = DataAugLoss(threshold=edge_budget)

        self.gnn.to(device)
        
        
        self.lr_scheduler = lr_scheduler
        if adapt_params=='edge_gnn':
            self.edge_gnn_optimizer = Adam(list(self.dataAug_gnn.parameters())+list(self.edge_linear.parameters()), lr=adapt_lr,weight_decay=5e-4)
        elif adapt_params=='edge_linear':
            self.edge_gnn_optimizer = Adam(self.edge_linear.parameters(), lr=adapt_lr,weight_decay=5e-4)
        else:
            assert "Please specify  correct 'adapt_params'!"
        

        self.optimizer = Adam(self.parameters(), lr=lr,weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=lr_decay, patience=patience, min_lr=1e-3)
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.metric_func = Accuracy(task='multiclass',num_classes=nclass,top_k=1).to(device) if valid_metric=='acc' else AUROC(task='binary').to(device)
        self.metric_name = valid_metric
        self.train_grad_sim = []
        self.val_grad_sim = []
        
        self.valid_metric_list = []
        self.meta_valid_metric_list = []
        self.best_valid_metric = -1.
        self.test_metric=-1.
        self.early_stop_epochs = early_stop_epochs
        self.epochs_since_improvement=0
        self.stop_training=False
        
        self.train_metrics = []
        self.val_metrics = []
        self.test_metrics = []


    def create_mlp(self,input_dim, hidden_dim, output_dim, num_layers,cls=True):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        if cls:
            layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)
    
    
    
    def learn_edge_weight(self, data,tau=0.2,return_probs = False):
        edge_index = data.edge_index
        X = self.dataAug_gnn(data.x,data.edge_index,edge_attr=data.edge_attr,batch=data.batch) # (N,F)
        s = X[edge_index[0]]
        t = X[edge_index[1]]
        edge_embeddings1 = torch.cat([s, t, s + t], dim=1)
        edge_embeddings2 = torch.cat([t, s, s + t], dim=1)
        edge_logits = self.edge_linear(edge_embeddings1+edge_embeddings2)
        # calc edge_prob and TV distance
        edge_probs = F.softmax(edge_logits, dim=1)[:,1]
        if return_probs:
            return edge_probs
        if self.calc_edge_stats:
            avg_causal_edge_probs,avg_causal_edge_rank = self.calc_average_prob_and_rank(edge_probs,data.edge_gt)
            self.edge_probs.append(avg_causal_edge_probs)
            self.edge_ranks.append(avg_causal_edge_rank)
        
        
        sorted_probs, _ = torch.sort(edge_probs)
        k = int(len(sorted_probs) * self.edge_prob_thres / 100.0)  # Calculate the number of lowest values to select
        sorted_probs = sorted_probs[:k]
        
        edge_tv_distance = total_variation_distance(sorted_probs)
        edge_weights = F.gumbel_softmax(edge_logits, tau=tau, hard=True, dim=1)[:,1]
        return edge_weights,edge_tv_distance



    def train_labelled_one_step(self,data):
        # self.encoder_model.train()
        data = data.to(self.device)
        y = data.y
        tot_edges = data.edge_index.shape[1]
        edge_weight,edge_tv_distance = self.learn_edge_weight(data) if self.useAutoAug else (None,None)
        # print (edge_weight)
        # print (self.useAutoAug)
        g = self.gnn(data.x, data.edge_index, batch=data.batch,edge_attr = data.edge_attr,edge_weight=edge_weight, return_both_rep=False)
        logits = self.cls_header(g)
        loss = self.ce_loss(logits, y.long().view(-1,))
        # if torch.isnan(loss).any():
        #     print ("loss nan")
        #     return -1
        # edge_reg = (torch.sum(edge_weight)/tot_edges-self.edge_budget)**2
        
        if self.useAutoAug:
            edge_reg = self.edge_loss(edge_weight.sum()/tot_edges)
            return loss + self.edge_penalty*edge_reg + self.edge_uniform_penalty*edge_tv_distance
        else:
            return loss

    def fit(self,dataloader,valid_dloader,test_dloader=None,epochs=50):
        if self.pe>0:
            state = deepcopy([self.useAutoAug,0,self.edge_uniform_penalty,self.penalty,self.edge_penalty])
            self.useAutoAug = False
            # self.encoder_model.learnable_aug=False
            self.epochs_since_improvement=0
            self.edge_uniform_penalty= 0.
            self.penalty = 0.
            self.edge_penalty = 0.
        for e in range(epochs):
            print (colored(f'Current Epoch {e}','red','on_yellow'))
            if e==self.pe and self.pe>0:
                self.useAutoAug,self.epochs_since_improvement,self.edge_uniform_penalty,self.penalty,self.edge_penalty = state
                self.best_valid_metric = 0.
            if self.stop_training:
                break
            total_losses = 0.
            steps = 0
            for data in dataloader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                loss = self.train_labelled_one_step(data)
                # print ('label loss:',labelled_loss.item(),'ssl loss:',ssl_loss.item()*self.penalty)
                if loss==-1: continue
                loss.backward()
                self.optimizer.step()
                # erm_losses += labelled_loss.item()
                # ssl_losses += ssl_loss.item()
                total_losses += loss.item()
                steps +=1
                # use colored to print three losses
            
            print (colored(f'Epoch {e}  Total Loss: {total_losses/steps}','red','on_white'))

            train_metric_score = self.evaluate_model(dataloader,'train',is_dataloader=True)
            val_metric_score = self.evaluate_model(valid_dloader,'valid',is_dataloader=True)
            test_metric_score= self.evaluate_model(test_dloader,'test',is_dataloader=True)
            self.train_metrics.append(train_metric_score)
            self.val_metrics.append(val_metric_score)
            self.test_metrics.append(test_metric_score)
            

            self.valid_metric_list.append((val_metric_score,test_metric_score))
            self.train()
    
    def evaluate_model(self, data_input,phase,is_dataloader=True,meta_loss=False,best_edge_weight=None):
        """
        For cls header evaluation
        Evaluate the model on a given dataset.

        Args:
        - dataloader: DataLoader for the dataset to evaluate.
        - phase: Phase of evaluation ('valid' or 'test') to control output messaging.

        Returns:
        - The average metric value across the dataset.
        - The average cosine similarity between SSL and ERM gradients.
        """

        self.eval()  # Set model to evaluation mode
        grads_sim = []
        logits_list = []
        labels_list = []
        steps = 0
        with torch.no_grad():
            if not is_dataloader:
                data_input = [data_input]
            for data in data_input:
                data = data.to(self.device)
                # Metric computation
                if best_edge_weight is not None:
                    edge_weight = best_edge_weight
                elif self.useAutoAug:
                    if self.num_samples==1:
                        edge_weight,_ = self.learn_edge_weight(data) if self.useAutoAug else None
                    else:
                        edge_weight_avg = []
                        for _ in range(self.num_samples):
                            edge_weight,_ = self.learn_edge_weight(data) if self.useAutoAug else None
                            edge_weight_avg.append(edge_weight.view(1,-1))
                        edge_weight = torch.cat(edge_weight_avg,dim=0).mean(dim=0)
                else:
                    edge_weight = None
                
                if self.featureMasking:
                    g = self.gnn(data.x*torch.sigmoid(self.featsMask), data.edge_index, batch=data.batch,edge_attr = data.edge_attr,edge_weight=edge_weight, return_both_rep=False)
                else:
                    g = self.gnn(data.x, data.edge_index, batch=data.batch,edge_attr = data.edge_attr,edge_weight=edge_weight, return_both_rep=False)
                logits = self.cls_header(g)
                if not is_dataloader:
                    self.train()
                    return logits
                
                logits_list.append(logits)
                labels_list.append(data.y.view(-1,))
                steps += 1
            
            all_logits = torch.cat(logits_list, dim=0)
            all_labels = torch.cat(labels_list, dim=0)
            
            # Compute metric with all logits and labels
            if self.metric_name=='acc':
                metric_score = self.metric_func(all_logits, all_labels).item()
            
            if self.metric_name=='auc':
                metric_score = self.metric_func(all_logits[:,1], all_labels).item() # use pos logits
            
            if phase.lower()=='valid' and self.lr_scheduler:
                self.scheduler.step(metric_score)
            
            if phase.lower()=='valid':
                print ('valid phase')
                if metric_score>self.best_valid_metric:
                    self.best_valid_metric = metric_score
                    self.epochs_since_improvement=0
                    # save model
                    self.best_states = deepcopy(self.state_dict())
                else:
                    self.epochs_since_improvement+=1
                    if self.epochs_since_improvement>=self.early_stop_epochs:
                        self.stop_training=True
                        print (colored(f'Early Stopping: No improvement for {self.early_stop_epochs} epochs','red','on_white'))
            if phase=='test':
                print(colored(f'{phase} Phase: Average {self.metric_name}: {metric_score}', 'blue','on_yellow'))
            else:
                print(colored(f'{phase} Phase: Average {self.metric_name}: {metric_score}', 'blue','on_white'))
            self.train()
            return metric_score

