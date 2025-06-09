import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch.nn.init as init


### GIN convolution along the graph structure
class GINConv(MessagePassing):

    def __init__(self, in_dim,emb_dim,edge_dim=-1):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
        # self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim),
        #                                torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim),torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU())
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        if edge_dim>1:
            self.edge_encoder = torch.nn.Linear(edge_dim, in_dim)
        self.edge_dim = edge_dim


    def forward(self, x, edge_index, edge_attr=None,edge_weight=None):
        if self.edge_dim > 1 and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        else:
            edge_attr = None
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x,edge_attr=edge_attr, edge_weight=edge_weight))
        return out

        
    def message(self, x_j,edge_attr, edge_weight):
        if edge_attr is not None and self.edge_dim>1 and x_j.shape[0]==edge_attr.shape[0]:
            x_j = x_j + edge_attr
            
        if edge_weight is not None:
            return F.relu(x_j * edge_weight.view(-1,1))
        return F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out
    


### GCN convolution along the graph structure
class GCNConv(MessagePassing):

    def __init__(self, in_dim,emb_dim, edge_dim=-1):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(in_dim, emb_dim)
        self.bias = Parameter(torch.empty(emb_dim))
        init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        init.zeros_(self.bias)
        self.edge_dim = edge_dim
        # print (in_dim,emb_dim,edge_dim)
        if edge_dim>0:
            self.edge_encoder = torch.nn.Linear(edge_dim, emb_dim)

    def forward(self, x, edge_index, edge_attr=None,edge_weight=None):
        if self.edge_dim <= 0:
            edge_embedding = None
        else:
            edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding,
                              norm=norm,edge_weight=edge_weight)+self.bias

    def message(self, x_j, edge_attr, norm,edge_weight):
        # assert not torch.isnan(x_j).any(), "Input x contains NaN"
        # assert not torch.isnan(edge_attr).any(), "Edge attributes contain NaN"
        if edge_weight is not None:
            assert not torch.isnan(edge_weight).any(), "Edge weights contain NaN"
        if self.edge_dim < 0:
            if edge_weight is None:
                return norm.view(-1, 1) * F.relu(x_j)
            else:
                return norm.view(-1, 1) * F.relu(x_j*edge_weight.view(-1,1))
        else:
            if edge_weight is None:
                return norm.view(-1, 1) * F.relu(x_j + edge_attr)
            else:
                return norm.view(-1, 1) * F.relu(x_j + edge_attr)*edge_weight.view(-1,1)

    def update(self, aggr_out):
        return aggr_out


