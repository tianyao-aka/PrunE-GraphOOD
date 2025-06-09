import json
import math
import os
import os.path as osp
import sys
import time
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import path
import shutil
# import time
import json
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch_geometric.loader import DataLoader
from torch_geometric import transforms as T
# # pytorch lightning
from torch_geometric.datasets import *
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import math
import argparse
import numpy as np
import higher
from termcolor import colored


from copy import deepcopy
from torch_geometric.loader import DataLoader
from termcolor import colored
from datetime import datetime
import networkx as nx
import scipy as sp
import pymetis
from tqdm import tqdm
import time
from torch_geometric.datasets import GNNBenchmarkDataset
import argparse
import warnings
from glob import glob


from drugood.datasets import build_dataset
from mmcv import Config
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset

from modelNew.prune import Model
from modelNew.utils import save_numpy_array_to_file,RandomEdgeDrop
from GOOD.data.good_datasets.good_cmnist import GOODCMNIST
from GOOD.data.good_datasets.good_motif import GOODMotif
from GOOD.data.good_datasets.good_hiv import GOODHIV
from GOOD.data.good_datasets.good_sst2 import GOODSST2


def write_results_to_file(fpath, n, s):
    # Check if the directory exists, if not, create it
    if not os.path.exists(fpath):
        try:
            os.makedirs(fpath)
        except:
            pass

    # Construct full file path
    full_path = os.path.join(fpath, n)

    # Open the file in write mode, which will create the file if it does not exist
    # and overwrite it if it does. Then write the string to the file.
    with open(full_path, 'w') as f:
        f.write(s)


def save_model_weights(dir_path, model,save_name = "model_weights.pt"):
    """
    Saves the weights of a PyTorch model to the specified directory.

    Parameters:
    - dir_path (str): The directory path where the model weights will be saved.
    - save_name (str): The name of the file to save the model weights (e.g., 'model_params.pt').
    - model (torch.nn.Module): The PyTorch model whose weights are to be saved.

    Returns:
    - None: The function saves the model weights to the specified location.
    """
    # Ensure the directory exists
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Full path for saving the model
    save_path = os.path.join(dir_path, save_name)
    
    # Save the model's state_dict (parameters)
    torch.save(model.state_dict(), save_path)
    
    print(f"Model weights saved to {save_path}")

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

# provide a parser for the command line
parser = argparse.ArgumentParser()
# add augument for string arguments


parser.add_argument('--dataset', type=str)
parser.add_argument('--root', default='./GOOD_data', type=str, help='directory for datasets.')

# Add arguments for model configuration. nfeats,nhid and nclass should be set manually
parser.add_argument('--nfeat', type=int, default=39, help='Number of features.') 
parser.add_argument('--nhid', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--nclass', type=int, default=2, help='Number of classes.')
parser.add_argument('--edge_dim', type=int, default=-1, help='dim of edge attr')
parser.add_argument('--nlayers', type=int, default=3, help='Number of GNN layers.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
parser.add_argument('--save_mem', action='store_true', default=True, help='Enable memory-saving mode.')
parser.add_argument('--jk', type=str, default='last', choices=['last', 'concat'], help='Jumping knowledge mode.')
parser.add_argument('--node_cls', action='store_true', default=False, help='node classification or graph cls')
parser.add_argument('--pooling', type=str, default='sum', choices=['sum', 'attention'], help='Pooling strategy.')
parser.add_argument('--with_bn', action='store_true', default=False, help='Enable batch normalization.')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay rate.')
parser.add_argument('--with_bias', action='store_true', default=True, help='Include bias in layers.')
parser.add_argument('--base_gnn', type=str, default='gin', choices=['gin', 'gcn'], help='Base GNN model type.')
parser.add_argument('--edge_gnn', type=str, default='gin', choices=['gin', 'gcn'], help='data aug GNN model type.')
parser.add_argument('--edge_gnn_layers', type=int, default=2, help='No. layers')
parser.add_argument('--edge_budget', type=float, default=0.55, help='edge budget for edge removal')
parser.add_argument('--num_samples', type=int, default=1, help='No. of samples of graphs')
parser.add_argument('--edge_penalty', type=float, default=0., help='penalty for regularization of data aug')
parser.add_argument('--edge_uniform_penalty', type=float, default=0., help='penalty for edge sampling uniformity penalty')
parser.add_argument('--edge_prob_thres', type=int, default=50, help='edge prob thres of k in int(50%)')

parser.add_argument('--penalty', type=float, default=1e-1, help='SSL Penalty weight.')
parser.add_argument('--gradMatching_penalty', type=float, default=1.0, help='meta cls Penalty weight.') #! actually meta cls penalty, not gradient matching penalty
parser.add_argument('--featureMasking', action='store_true', default=False, help='mask input or not')
parser.add_argument('--useAutoAug', action='store_true', default=True, help='use learnable edge dropping')
parser.add_argument('--uniform_p', type=float, default=-1., help='prob for uniform reg')


parser.add_argument('--shift', type=str, default='covariate')
# parser.add_argument('--useAutoAug', action='store_true', default=False, help='use learnable edge dropping')
# Add arguments for training configuration

parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--adapt_lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--lr_scheduler', action='store_false', default=True, help='Enable learning rate scheduler.')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--early_stop_epochs', type=int, default=10)
parser.add_argument('--pretraining_epochs', type=int, default=10)
parser.add_argument('--patience', type=int, default=50, help='Patience for learning rate scheduler.')
parser.add_argument('--lr_decay', type=float, default=0.75, help='Learning rate decay factor.')
parser.add_argument('--project_layer_num', type=int, default=2, help='Number of projection layers.')
parser.add_argument('--temp', type=float, default=0.2, help='Temperature parameter for contrastive loss.')
parser.add_argument('--valid_metric', type=str, default='auc', help='Validation metric.')
parser.add_argument('--domain', type=str, default='scaffold', help='Validation metric.')
parser.add_argument('--mol_encoder', action='store_true', default=False, help='Use AtomEncoder or not')
parser.add_argument('--use_vGIN', action='store_true', default=False, help='Use virtual GIN or not')

#! use RSC algo
parser.add_argument('--use_rsc', action='store_true', default=False, help='Use RFC algo or not')
parser.add_argument('--p_rsc', type=float, default=0.3, help='p params in rf algo')


#! use div cls
parser.add_argument('--use_div_cls', action='store_true', default=False, help='Use divCls algo or not')
parser.add_argument('--div_headers', type=int, default=10, help='#headers')
parser.add_argument('--div_reg', type=float, default=1e-1, help='div reg')


parser.add_argument('--erm', action='store_true', default=False, help='use erm')
parser.add_argument('--fname_str', type=str, default='', help='additional name for folder name')

parser.add_argument('--addRandomFeature', action='store_true', default=False, help='add random features for SPMotif datasets')

#! EdgeDrop
parser.add_argument('--random_edge_drop', type=float, default=0.0)

# System configuration
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--seed', type=int, default=1) 
parser.add_argument('--debug', action='store_true', default=False, help='Enable memory-saving mode.')
# Parse arguments

args = parser.parse_args()

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Construct result directory name

result_fname = f"{args.dataset}/"

print ('result name:',result_fname)

if args.fname_str != "":
    result_fname += f"{args.fname_str}/{args.domain}/"
if not args.useAutoAug:
    result_fname += f"ERM/{args.base_gnn}_vGIN_{args.use_vGIN}_nhid_{args.nhid}_nlayers_{args.nlayers}_dropout_{args.dropout}"
else:
    result_fname += f"autoAug_{args.useAutoAug}/{args.base_gnn}_vGIN_{args.use_vGIN}_nhid_{args.nhid}_nlayers_{args.nlayers}_uniformP_{args.uniform_p}"
result_fname += f"_saveMem_{args.save_mem}_jk_{args.jk}_withBN_{args.with_bn}"
if args.use_rsc:
    result_fname += f"_rscP_{args.p_rsc}"
if args.use_div_cls:
    result_fname += f"_divHeaders_{args.div_headers}_divReg_{args.div_reg}"
    
result_fname += f"_edgeGNN_{args.edge_gnn}_egLayers_{args.edge_gnn_layers}_edgeBudget_{args.edge_budget}_edgeUniformPenalty_{args.edge_uniform_penalty}_edgePenalty_{args.edge_penalty}_edgeProbThres_{args.edge_prob_thres}_numSamples_{args.num_samples}_seed_{args.seed}/"

#! DropEdge

if args.random_edge_drop>0:
    randDrop = RandomEdgeDrop(args.random_edge_drop)
else:
    randDrop = None



args = vars(args)

dataset_name = args['dataset']
s = time.time()


args['device'] = 'cpu' if not torch.cuda.is_available() else args['device']
# args['device'] = 'cpu'
print ('using device:',args['device'])


dataset_name = args["dataset"]
args['dataset_name'] = dataset_name


#! Init dataset

workers = 2 if torch.cuda.is_available() else 0
if 'cmnist' in args['dataset'].lower():
    args['nclass'] = 10
    args["nfeat"] = 3
    args['nlayers']=3
    metric_name = 'acc'
    dataset, meta_info = GOODCMNIST.load(args['root'], domain=args['domain'], shift=args['shift'], generate=False)
    
    train_dataset = dataset["train"]
    train_dataset.transform = randDrop
    
    train_loader = DataLoader(train_dataset,batch_size=args["batch_size"],num_workers=workers,shuffle=True)
    valid_loader = DataLoader(dataset["val"],batch_size=args["batch_size"],shuffle=False,num_workers=workers)
    test_loader = DataLoader(dataset["test"],batch_size=args["batch_size"],shuffle=False,num_workers=workers)
    args['valid_metric'] = metric_name

elif 'motif' in args['dataset'].lower():
    #drugood_lbap_core_ic50_assay.json
    args['nclass'] = 3
    args["nfeat"] = 1
    args['nlayers']=3
    metric_name='acc'
    args['valid_metric'] = metric_name
    dataset, meta_info = GOODMotif.load(args['root'], domain=args['domain'], shift=args['shift'], generate=False)
    
    train_dataset = dataset["train"]
    train_dataset.transform = randDrop
    
    train_loader = DataLoader(train_dataset,batch_size=args["batch_size"],num_workers=workers,shuffle=True)
    valid_loader = DataLoader(dataset["val"],batch_size=args["batch_size"],shuffle=False,num_workers=workers)
    test_loader = DataLoader(dataset["test"],batch_size=args["batch_size"],shuffle=False,num_workers=workers)
    
elif 'goodhiv' in args['dataset'].lower():
    args['nclass'] = 2
    metric_name='auc'
    args['valid_metric'] = metric_name
    dataset, meta_info = GOODHIV.load(args['root'], domain=args['domain'], shift=args['shift'], generate=False)

    train_dataset = dataset["train"]
    train_dataset.transform = randDrop

    train_loader = DataLoader(train_dataset,batch_size=args["batch_size"],num_workers=workers,shuffle=True)
    valid_loader = DataLoader(dataset["val"],batch_size=args["batch_size"],shuffle=False,num_workers=workers)
    test_loader = DataLoader(dataset["test"],batch_size=args["batch_size"],shuffle=False,num_workers=workers)

elif 'bbbp' in args['dataset'].lower() or 'bace' in args['dataset'].lower():
    name = 'ogbg-molbbbp' if 'bbbp' in args['dataset'].lower() else 'ogbg-molbace'
    print (f"use dataset: {name}")
    dataset = PygGraphPropPredDataset(name=name, root=args['root'])
    args['nclass'] = 2
    metric_name='auc'
    args['valid_metric'] = metric_name
    if args['domain'] == "scaffold":
        split_idx = dataset.get_idx_split()
    else:
        split_idx = size_split_idx(dataset)
    
    train_dataset = dataset[split_idx["train"]]
    train_dataset.transform = randDrop
    
    train_loader = DataLoader(train_dataset,batch_size=args["batch_size"],num_workers=workers,shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]],batch_size=args["batch_size"],shuffle=False,num_workers=workers)
    test_loader = DataLoader(dataset[split_idx["test"]],batch_size=args["batch_size"],shuffle=False,num_workers=workers)


else:
    raise Exception("Invalid dataset name")

# log
datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")

#! init model
args['dataset_name'] = args['dataset']
model = Model(**args)
model.to(args['device'])


  
model.fit(train_loader,valid_loader,test_loader,epochs=args["epochs"])
model.load_state_dict(model.best_states)
res = model.valid_metric_list
res = sorted(res,key = lambda x:x[0],reverse=True)   # get the test perf according t best validation perf
val_score,test_score = res[0]


res = np.array([val_score,test_score])

print (f'Best valid perf:{res[0]}, Test perf accordingly:{res[1]}')



