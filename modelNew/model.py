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


import os.path as osp
import GCL.losses as L
import GCL.augmentors as A

from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader

from torchmetrics import AUROC,Accuracy
import numpy as np
import pandas as pd
from termcolor import colored


from modelNew.base_model import BaseModel
from modelNew.ssl_module import Encoder
from modelNew.gcn import GCN
from modelNew.gin import GIN
from modelNew.utils import *
# try:
#     from modelNew.base_model import BaseModel
#     from modelNew.ssl_module import Encoder
#     from modelNew.gcn import GCN
#     from modelNew.gin import GIN
#     from modelNew.utils import *
# except:
#     print ('import from local')
#     from base_model import BaseModel
#     from gcn import GCN
#     from gin import GIN
#     from ssl_module import Encoder
#     from utils import *


class Model(BaseModel):
    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5,save_mem=True,jk='last',node_cls=False,pooling='sum',
                with_bn=False, weight_decay=5e-6,lr=1e-3,lr_scheduler=True,patience=50,early_stop_epochs=50,lr_decay=0.75,penalty=0.1,project_layer_num=2,edge_gnn=None,temp=0.2, with_bias=True,base_gnn='gin',valid_metric='acc', device='cpu', **args):

        super(Model, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.jk = jk
        self.node_cls = node_cls
        self.cls_header = self.create_mlp(nhid,nhid,nclass,project_layer_num)
        self.ssl_header = self.create_mlp(nhid,nhid,nclass,project_layer_num,cls=False)
        self.penalty = penalty
        
        if base_gnn=='gin':
            self.gnn = GIN(nfeat, nhid, nclass, nlayers, dropout=dropout,jk=jk,node_cls=node_cls,pooling=pooling,
                with_bn=with_bn, weight_decay=weight_decay)
        if base_gnn=='gcn':
            self.gnn = GCN(nfeat, nhid,nclass, nlayers=nlayers, dropout=dropout,save_mem=save_mem,jk=jk,node_cls=node_cls,pooling=pooling,
                with_bn=with_bn, weight_decay=weight_decay, with_bias=with_bias)
            
        
        if edge_gnn=='gin':
            self.dataAug_gnn = GIN(nfeat, nhid, nclass, nlayers=2, dropout=dropout,jk='last',node_cls=True,
                with_bn=with_bn, weight_decay=weight_decay)
        if edge_gnn=='gcn':
            self.dataAug_gnn = GCN(nfeat, nhid,nclass, nlayers=2, dropout=dropout,save_mem=save_mem,jk='last',node_cls=True,
                with_bn=with_bn, weight_decay=weight_decay, with_bias=with_bias)
        if edge_gnn is not None:
            self.edge_linear = nn.Sequential(nn.Linear(3*nhid, 2*nhid),nn.ReLU(),nn.Linear(2*nhid, 2)) 
        
        self.gnn.to(device)
        aug1 = A.Identity()
        aug2 = A.EdgeRemoving(pe=0.1)
        self.encoder_model = Encoder(self.gnn, [aug1,aug2]).to(device)
        self.contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=temp), mode='G2G').to(device)
        
        
        self.lr_scheduler = lr_scheduler
        self.optimizer = Adam(self.parameters(), lr=lr,weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=lr_decay, patience=patience, min_lr=1e-3)

        self.loss_func = nn.CrossEntropyLoss()
        self.metric_func = Accuracy(task='multiclass',num_classes=nclass).to(device) if valid_metric=='acc' else AUROC(task='binary').to(device)
        self.metric_name = valid_metric
        self.train_grad_sim = []
        self.val_grad_sim = []
        
        
        self.best_valid_metric = -1.
        self.test_metric=-1.
        self.early_stop_epochs = early_stop_epochs
        self.epochs_since_improvement=0
        self.stop_training=False
        
        
        self.train_metrics = []
        self.val_metrics = []
        self.test_metrics = []
        self.train_grad_sim = []
        self.val_grad_sim = []
        self.test_grad_sim = []


    def create_mlp(self,input_dim, hidden_dim, output_dim, num_layers,cls=True):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        if cls:
            layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)
    
    
    def learn_edge_weight(self, data,tau=0.2):
        edge_index = data.edge_index
        X = self.dataAug_gnn(data.x,data.edge_index) # (N,F)
        s = X[edge_index[0]]
        t = X[edge_index[1]]
        edge_embeddings1 = torch.cat([s, t, s + t], dim=1)
        edge_embeddings2 = torch.cat([t, s, s + t], dim=1)
        edge_logits = self.edge_linear(edge_embeddings1+edge_embeddings2)
        edge_weights = F.gumbel_softmax(edge_logits, tau=tau, hard=True, dim=1)[:,1]
        return edge_weights
        
        
    
    def train_ssl_one_step(self,data):
        # self.encoder_model.train()
        # data = data.to(self.device)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, _, _, _, g1, g2 = self.encoder_model(data.x, data.edge_index, data.batch)
        g1, g2 = [self.ssl_header(g) for g in [g1, g2]]
        loss = self.contrast_model(g1=g1, g2=g2, batch=data.batch)
        return loss
    
    def train_labelled_one_step(self,data):
        # self.encoder_model.train()
        # data = data.to(self.device)
        y = data.y
        g = self.gnn(data.x, data.edge_index, batch=data.batch, return_both_rep=False)
        logits = self.cls_header(g)
        loss = self.loss_func(logits, y)
        return loss


    def fit(self,dataloader,valid_dloader,test_dloader=None,epochs=50):
        for e in range(epochs):
            print (colored(f'Current Epoch {e}','red','on_yellow'))
            
            if self.stop_training:
                break
            erm_losses = 0.
            ssl_losses = 0.
            total_losses = 0.
            steps = 0
            for data in dataloader:
                self.optimizer.zero_grad()
                ssl_loss = self.train_ssl_one_step(data)
                labelled_loss = self.train_labelled_one_step(data)
                # do sth to get the gradients
                loss = self.penalty*ssl_loss + 0.*labelled_loss
                print ('label loss:',labelled_loss.item(),'ssl loss:',ssl_loss.item())
                loss.backward()
                self.optimizer.step()
                erm_losses += labelled_loss.item()
                ssl_losses += ssl_loss.item()
                total_losses += loss.item()
                steps +=1
                # use colored to print three losses
            
            print (colored(f'Epoch {e} SSL Loss: {ssl_losses/steps} Labelled Loss: {erm_losses/steps} Total Loss: {total_losses/steps}','red','on_white'))
            train_metric_score, train_avg_grad_sim = self.evaluate_model(dataloader,'train')
            val_metric_score, val_avg_grad_sim = self.evaluate_model(valid_dloader,'valid')
            self.train_metrics.append(train_metric_score)
            self.val_metrics.append(val_metric_score)
            self.train_grad_sim.append(train_avg_grad_sim)
            self.val_grad_sim.append(val_avg_grad_sim)
            if test_dloader is not None:
                test_metric_score, test_avg_grad_sim = self.evaluate_model(test_dloader,'test')
                self.test_metrics.append(test_metric_score)
                self.test_grad_sim.append(test_avg_grad_sim)
            self.train()

    
    def evaluate_model(self, dataloader,phase):
        """
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

        for data in dataloader:
            # Forward passes to compute losses
            ssl_loss = self.train_ssl_one_step(data)
            labelled_loss = self.train_labelled_one_step(data)
            
            # Metric computation
            logits = self.cls_header(self.gnn(data.x, data.edge_index, batch=data.batch, return_both_rep=False))
            logits_list.append(logits)
            labels_list.append(data.y.view(-1,))
            
            # Enable gradients temporarily for gradient similarity calculation
            ssl_loss.backward(retain_graph=True)
            ssl_gradients = get_model_gradients_vector(self.gnn).view(1, -1)
            self.optimizer.zero_grad()  # Reset gradients
            
            labelled_loss.backward(retain_graph=True)
            labelled_gradients = get_model_gradients_vector(self.gnn).view(1, -1)
            self.optimizer.zero_grad()  # Reset gradients again
            
            grads_sim.append(F.cosine_similarity(ssl_gradients, labelled_gradients).item())
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
            if metric_score>self.best_valid_metric:
                self.best_valid_metric = metric_score
                self.epochs_since_improvement=0
            else:
                self.epochs_since_improvement+=1
                if self.epochs_since_improvement>=self.early_stop_epochs:
                    self.stop_training=True
                    print (colored(f'Early Stopping: No improvement for {self.early_stop_epochs} epochs','red','on_white'))
        
        average_grad_sim = np.mean(grads_sim)
        print(colored(f'{phase} Phase: Average {self.metric_name}: {metric_score}, Average Grad Sim: {average_grad_sim}', 'blue','on_white'))

        
        
        return metric_score, average_grad_sim
    
    