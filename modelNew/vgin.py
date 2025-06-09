from operator import is_
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv
from termcolor import colored

import sys

try:
    from base_model import BaseModel
    from conv import GINConv
except:
    from modelNew.base_model import BaseModel
    from modelNew.conv import GINConv
    
from torch_sparse import coalesce, SparseTensor
import torch.optim as optim
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import global_add_pool,global_mean_pool,AttentionalAggregation
# from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from torch import Tensor
from GOOD.utils.data import x_map, e_map
import pdb

class AtomEncoder(torch.nn.Module):
    r"""
    atom (node) feature encoding specified for molecule data.

    Args:
        nhid: number of dimensions of embedding
    """

    def __init__(self, nhid):

        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        feat_dims = list(map(len, x_map.values()))

        for i, dim in enumerate(feat_dims):
            emb = torch.nn.Embedding(dim, nhid)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        r"""
        atom (node) feature encoding specified for molecule data.

        Args:
            x (Tensor): node features

        Returns (Tensor):
            atom (node) embeddings
        """
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(torch.nn.Module):
    r"""
    bond (edge) feature encoding specified for molecule data.

    Args:
        nhid: number of dimensions of embedding
    """

    def __init__(self, nhid):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        edge_feat_dims = list(map(len, e_map.values()))

        for i, dim in enumerate(edge_feat_dims):
            emb = torch.nn.Embedding(dim, nhid)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        r"""
        bond (edge) feature encoding specified for molecule data.

        Args:
            edge_attr (Tensor): edge attributes

        Returns (Tensor):
            bond (edge) embeddings

        """
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding


class vGIN(BaseModel):
    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.0,edge_dim=-1,jk='last',node_cls=False,pooling='sum',
                with_bn=False, weight_decay=5e-6, with_bias=False, **args):

        super(vGIN, self).__init__()
        
        self.args = args
        dataset_name = args.get('dataset_name')
        mol_encoder = args.get('mol_encoder',False)
        self.is_ogbg = True if 'ogbg' in dataset_name.lower() else False
        self.is_ogbg = True if mol_encoder else False
        print (colored(f'use MolEncoder: {self.is_ogbg}','yellow'))
        self.jk = jk
        self.node_cls = node_cls
        self.edge_dim = edge_dim
        self.layers = nn.ModuleList([])
        if self.is_ogbg and edge_dim>0:
            self.node_encoder = AtomEncoder(nhid=nhid)
            self.edge_encoder = BondEncoder(nhid=nhid)
        
        if with_bn:
            self.bns = nn.ModuleList()
        
        if nlayers == 1:
            self.layers.append(GINConv(nfeat, nhid,edge_dim=edge_dim))
        else:
            # Initialize GINConv layers
            if self.is_ogbg and edge_dim>0:
                self.layers.append(GINConv(nhid, nhid,edge_dim=nhid))
            else:
                self.layers.append(GINConv(nfeat, nhid,edge_dim=edge_dim))
            if with_bn:
                self.bns.append(nn.BatchNorm1d(nhid))
            for _ in range(1, nlayers):
                if self.is_ogbg and edge_dim>0:
                    self.layers.append(GINConv(nhid, nhid,edge_dim=nhid))
                else:
                    self.layers.append(GINConv(nhid, nhid,edge_dim=edge_dim))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
                if self.jk=='last':
                    pass
                else:
                    self.jk_layer = nn.Linear(nhid * nlayers, nhid)



        self.dropout = dropout
        self.weight_decay = weight_decay
        self.output = None
        self.best_model = None
        self.best_output = None
        self.with_bn = with_bn
        self.name = 'GIN'
        if pooling=='sum':
            self.pool = global_add_pool
        if pooling=='mean':
            self.pool = global_mean_pool
        if pooling=='attention':
            self.pool = AttentionalAggregation(gate_nn=nn.Linear(nhid,1))
            
            
        # virtual node
        self.virtual_node_embedding = nn.Embedding(1, nhid)
        self.virtual_mlp = nn.Sequential(*(
                [nn.Linear(nhid, 2 * nhid),
                 nn.BatchNorm1d(2 * nhid), nn.ReLU()] +
                [nn.Linear(2 * nhid, nhid),
                 nn.BatchNorm1d(nhid), nn.ReLU(),
                 nn.Dropout()]
        ))
        self.virtual_pool = global_add_pool
        

    def forward(self, x, edge_index, edge_attr = None,edge_weight=None,batch=None,return_both_rep=False):
        xs = []
        assert batch is not None, "Please specify 'batch' for graph pooling!"
        x, edge_index, edge_attr,edge_weight = self._ensure_contiguousness(x, edge_index, edge_attr,edge_weight)
        # if edge_weight is not None:
        #     adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2 * x.shape[:1]).t()
        
        
        virtual_node_feat = self.virtual_node_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
        if self.is_ogbg and self.edge_dim>0:
            x = self.node_encoder(x.long())
            if edge_attr is not None:
                edge_attr = self.edge_encoder(edge_attr.long())

        for ii, layer in enumerate(self.layers):
            if edge_weight is not None:
                x = layer(x, edge_index,edge_attr,edge_weight)
            else:
                # x = layer(x, edge_index, edge_weight=edge_weight)
                
                x = layer(x, edge_index,edge_attr)
            
            x = x + virtual_node_feat[batch]
            if ii != len(self.layers):
                if self.with_bn:
                    x = self.bns[ii](x)
                if self.dropout>0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                virtual_node_feat = self.virtual_mlp(self.virtual_pool(x, batch) + virtual_node_feat)
            if self.jk == 'concat':
                xs.append(x)
        
        
        
        if self.jk=='last':
            if self.node_cls:
                return x
            else:
                g = self.pool(x,batch)
                return (x,g) if return_both_rep else g # return node rep then graph rep
        else:
            x = torch.cat(xs, dim=1)
            x = self.jk_layer(x)
            if self.node_cls:
                return x
            else:
                g = self.pool(x,batch)
                return (x,g) if return_both_rep else g # return node rep then graph rep
        # return F.log_softmax(x, dim=1)


    def initialize(self):
        for m in self.layers:
            m.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

