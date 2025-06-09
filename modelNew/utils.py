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
from torch_geometric.utils import k_hop_subgraph

from torch_geometric.transforms import GDC

import os.path as osp
import GCL.losses as L
import GCL.augmentors as A

from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader,Data, Batch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.transforms import BaseTransform
import os
import random

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

class RandomAugmentationComposer:
    def __init__(
        self,
        node_drop_prob=0.1,
        edge_drop_prob=0.1,
        ppr_alpha=0.1,
        ppr_eps=1e-4,
        use_cache=True,
        add_self_loop=True,
        p=[0.,1.,0.]
    ):
        
        """
        Initialize the RandomAugmentationComposer with specified augmentation probabilities and parameters.
        
        Args:
            node_drop_prob (float): Probability of dropping each node in RandomNodeDrop. Default is 0.0.
            edge_drop_prob (float): Probability of dropping each edge in RandomEdgeDrop. Default is 0.0.
            ppr_alpha (float): Teleportation probability in PPR computation for PPRDiffusion. Default is 0.2.
            ppr_eps (float): Tolerance for convergence in PPR computation for PPRDiffusion. Default is 1e-4.
            use_cache (bool): Whether to cache the transformed data in PPRDiffusion. Default is True.
            add_self_loop (bool): Whether to add self-loops in PPRDiffusion. Default is True.
            p (list of floats): Probabilities for selecting each augmentation method. Should sum to 1.
                                Default is uniform probabilities for the available methods.
        """
        # Initialize augmentation methods with their respective parameters
        self.augmentations = [
            RandomNodeDrop(drop_prob=node_drop_prob),
            RandomEdgeDrop(drop_prob=edge_drop_prob),
            PPRDiffusion(alpha=ppr_alpha, eps=ppr_eps, use_cache=use_cache, add_self_loop=add_self_loop)
        ]
        
        self.num_augmentations = len(self.augmentations)
        
        # Set probabilities p
        if p is None:
            # Default to uniform probabilities
            self.p = [1.0 / self.num_augmentations] * self.num_augmentations
        else:
            assert len(p) == self.num_augmentations, "Length of probability list p must match number of augmentations."
            assert abs(sum(p) - 1.0) < 1e-6, "Probabilities in p must sum to 1."
            self.p = p

    def __call__(self, data):
        """
        Apply one of the augmentation methods to the data, selected randomly according to probabilities p.
        
        Args:
            data (torch_geometric.data.Data): The input data object.
        
        Returns:
            torch_geometric.data.Data: The augmented data object.
        """
        # Randomly select one augmentation method according to probabilities p
        idx = np.random.choice(self.num_augmentations, p=self.p)
        augmentation = self.augmentations[idx]
        
        # Apply the selected augmentation method to the data
        data_augmented = augmentation(data)
        
        return data_augmented


class RandomNodeDrop:
    def __init__(self, drop_prob):
        """
        Initialize the transformation with the probability of dropping a node.
        
        Args:
            drop_prob (float): The probability of dropping each node. Should be between 0 and 1.
        """
        assert 0 <= drop_prob <= 1, "drop_prob must be between 0 and 1."
        self.drop_prob = drop_prob

    def __call__(self, data):
        """
        Apply the random node drop transformation to a PyG Data object by setting
        the features of randomly selected nodes to zero.
        
        Args:
            data (torch_geometric.data.Data): The input data object with node features 'x'.
        
        Returns:
            torch_geometric.data.Data: The transformed data object with some node features set to zero.
        """
        x = data.x  # Node features

        if x is None:
            raise ValueError("Data object must have node features 'x' to apply RandomNodeDrop.")

        # Determine which nodes to keep
        num_nodes = x.size(0)
        keep_mask = torch.rand(num_nodes) > self.drop_prob  # Boolean mask indicating nodes to keep

        # Set features of dropped nodes to zero
        x[~keep_mask] = 0

        # Update the data object with modified node features
        data.x = x
        
        return data


class PPRDiffusion:
    def __init__(self, alpha=0.1, eps=1e-4, use_cache=True, add_self_loop=True):
        """
        Initialize the PPRDiffusion transformation.

        Args:
            alpha (float): The teleportation probability in the PPR computation (default: 0.2).
            eps (float): The tolerance for convergence in the PPR computation (default: 1e-4).
            use_cache (bool): Whether to cache the transformed data to avoid recomputation (default: True).
            add_self_loop (bool): Whether to add self-loops to the graph before PPR computation (default: True).
        """
        
        assert 0 < alpha < 1, "'alpha' must be between 0 and 1."
        assert eps > 0, "'eps' must be a positive value."
        self.alpha = alpha
        self.eps = eps
        self.use_cache = use_cache
        self.add_self_loop = add_self_loop
        self._cache = {}

        # Initialize the GDC transform with PPR diffusion
        self.gdc = GDC(
            self_loop_weight=1.0 if self.add_self_loop else 0.0,
            normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=dict(method='ppr', alpha=self.alpha),
            sparsification_kwargs=dict(method='threshold', eps=self.eps),
            exact=True,
        )

    def __call__(self, data):
        """
        Apply the PPR diffusion transformation to a PyG Data or Batch object.

        Args:
            data (torch_geometric.data.Data or torch_geometric.data.Batch): The input data object containing 'edge_index' (and optionally 'edge_weight').

        Returns:
            torch_geometric.data.Data or torch_geometric.data.Batch: The transformed data object with edges updated based on PPR diffusion.
        """
        # Check if data is a Batch of graphs
        if isinstance(data, Batch):
            # Convert Batch to list of Data objects
            data_list = data.to_data_list()
            transformed_data_list = []

            for graph_data in data_list:
                # Check the consistency of edge indices and features
                try:
                    # Apply GDC to each individual graph
                    transformed_graph = self.gdc(graph_data)
                    transformed_data_list.append(transformed_graph)
                except RuntimeError as e:
                    print(f"Error processing graph: {e}")
                    print(f"Graph edge_index size: {graph_data.edge_index.size()}")
                    print(f"Graph node feature size: {graph_data.x.size()}")
                    continue

            # Re-batch the transformed graphs
            data_transformed = Batch.from_data_list(transformed_data_list)

        else:
            # Apply the GDC transformation to compute the PPR diffused graph
            try:
                data_transformed = self.gdc(data)
            except RuntimeError as e:
                print(f"Error processing data: {e}")
                print(f"Data edge_index size: {data.edge_index.size()}")
                print(f"Data node feature size: {data.x.size()}")
                return data  # Return the original data if the transformation fails

        return data_transformed


class RandomEdgeDrop:
    def __init__(self, drop_prob):
        """
        Initialize the transformation with the probability of dropping an edge.
        
        Args:
            drop_prob (float): The probability of dropping each edge. Should be between 0 and 1.
        """
        assert 0 <= drop_prob <= 1, "drop_prob must be between 0 and 1."
        self.drop_prob = drop_prob

    def __call__(self, data):
        """
        Apply the random edge drop transformation to a PyG Data object.
        
        Args:
            data (torch_geometric.data.Data): The input data object with an edge_index.
        
        Returns:
            torch_geometric.data.Data: The transformed data object with some edges randomly dropped.
        """
        edge_index = data.edge_index

        # Determine which edges to keep
        num_edges = edge_index.size(1)
        keep_mask = torch.rand(num_edges) > self.drop_prob

        # Apply the mask to the edge_index
        edge_index = edge_index[:, keep_mask]

        # Create a new Data object with the modified edge_index
        data.edge_index = edge_index
        
        return data

def get_model_gradients_vector(model, loss_val,retain_graph=False):
    """
    Compute gradients of the given loss value with respect to all parameters in the PyTorch model,
    and concatenate these gradients into a single vector.

    Parameters:
    - model: The PyTorch model with respect to whose parameters the gradients will be computed.
    - loss_val: The scalar loss value for which gradients are to be computed.

    Returns:
    A single tensor vector containing all the gradients concatenated.
    """
    
    # Ensure the model's parameters are ready for gradient computation
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Compute gradients of loss_val with respect to model parameters
    grads = torch.autograd.grad(loss_val, params,retain_graph=retain_graph)
    
    # Flatten and concatenate all gradients into a single vector
    gradients_vector = torch.cat([grad.view(-1) for grad in grads])
    
    return gradients_vector



class DataAugLoss(nn.Module):
    def __init__(self, threshold=0.5, high_penalty=4.0, low_penalty=1.0):
        super(DataAugLoss, self).__init__()
        self.threshold = threshold
        self.high_penalty = high_penalty
        self.low_penalty = low_penalty

    def forward(self, input):
        # a = torch.sum(inputs==0)
        # b = torch.sum(inputs==1)
        # c = len(inputs)
        # print (a,b,c)
        # Ensure inputs are in the right shape and compute the condition

        # Compute losses for both conditions
        if input >= self.threshold:
            loss = self.high_penalty * (input - self.threshold)
        else:
            loss = self.low_penalty * (self.threshold - input)
        return loss
    

import torch
from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs) -> torch.FloatTensor:
        pass

    def __call__(self, anchor, sample, pos_mask=None, neg_mask=None, *args, **kwargs) -> torch.FloatTensor:
        loss = self.compute(anchor, sample, pos_mask, neg_mask, *args, **kwargs)
        return loss


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()

# class modInfoNCE(Loss):
#     def __init__(self, tau):
#         super(modInfoNCE, self).__init__()
#         self.tau = tau

#     def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
#         sim = _similarity(anchor, sample) / self.tau
#         exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
#         log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
#         loss = log_prob * pos_mask
#         loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
#         return -loss
    


class modInfoNCE(nn.Module):
    def __init__(self, tau):
        super(modInfoNCE, self).__init__()
        self.tau = tau

    def forward(self, anchor, sample, pos_mask, neg_mask):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss  # Returning mean loss to be consistent with PyTorch loss functions


class WrapperModel(nn.Module):
    def __init__(self, *models):
        super(WrapperModel, self).__init__()
        self.dataAug_gnn = models[0]
        self.edge_linear = models[1]
        self.gnn = models[2]
        self.encoder_model = models[3]
        self.contrast_model_non_agg = models[4]
        # self.meta_loss_mlp = models[5]
        self.ssl_header = models[5]
        self.cls_header = models[6]
        self.featsMask = models[7]
        self.meta_loss_mlp = models[8]
    def forward(self):
        pass
    
    
def check_grad(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Parameter '{name}' has gradients.")
        else:
            print(f"Parameter '{name}' does not have gradients.")
            
            
def compare_model_params(model1: nn.Module, model2: nn.Module, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """
    Compare parameters of two models to check if they are close enough.

    Parameters:
    - model1 (nn.Module): The first model to compare.
    - model2 (nn.Module): The second model to compare.
    - rtol (float): The relative tolerance parameter (default: 1e-05).
    - atol (float): The absolute tolerance parameter (default: 1e-08).

    Returns:
    - bool: True if all parameters of the two models are close enough, False otherwise.
    """
    # Extract parameters from both models and flatten them into tensors
    params1 = torch.cat([p.view(-1) for p in model1.parameters()])
    params2 = torch.cat([p.view(-1) for p in model2.parameters()])

    # Check if the flattened parameter tensors are close enough
    return torch.allclose(params1, params2, rtol=rtol, atol=atol)



def show_model_gradients(model):
    """
    Prints the gradients of all parameters in a PyTorch model for every module.
    
    Parameters:
    - model (nn.Module): The model whose gradients are to be displayed.
    """
    for module_name, module in model.named_modules():
        print(f"Module: {module_name} ({module.__class__.__name__})")
        for param_name, param in module.named_parameters(recurse=True):
            if param.grad is not None:
                print(f"  Param: {param_name}, Grad: {torch.sum(param.grad**2)}")
            else:
                print(f"  Param: {param_name}, Grad: None")


    
def save_numpy_array_to_file(array, file_path, file_name):
    """
    Save a NumPy array to a specified file path and file name.
    
    Parameters:
    - array (np.ndarray): The NumPy array to be saved.
    - file_path (str): The directory where the file will be saved.
    - file_name (str): The name of the file (without extension).
    
    Output:
    - None: The function saves the array to a .npy file at the specified location.
    """
    
    # Create the directory if it does not exist
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    # Full path to the file
    full_file_path = os.path.join(file_path, f"{file_name}.npy")
    
    # Delete the file if it already exists
    if os.path.exists(full_file_path):
        os.remove(full_file_path)
    
    # Save the NumPy array to the file
    np.save(full_file_path, array)
    

def save_tensor_to_file(tensor, file_path, file_name):
    """
    Save a PyTorch tensor or a list of tensors to a specified file path and file name.
    
    Parameters:
    - tensor (torch.Tensor or list of torch.Tensor): The tensor or list of tensors to be saved.
    - file_path (str): The directory where the file will be saved.
    - file_name (str): The name of the file (without extension).
    
    Output:
    - None: The function saves the tensor(s) to a .pt file at the specified location.
    """
    
    # Create the directory if it does not exist
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    # Full path to the file
    full_file_path = os.path.join(file_path, f"{file_name}.pt")
    
    # Delete the file if it already exists
    if os.path.exists(full_file_path):
        os.remove(full_file_path)
    
    # Save the tensor or list of tensors to the file
    torch.save(tensor, full_file_path)


def save_models_to_file(obj, file_path, file_name):
    """
    Save a PyTorch object (tensor, model, or list of models) to a specified file path and file name.
    
    Parameters:
    - obj (torch.Tensor, nn.Module, or list): The object (tensor, model, or list of models) to be saved.
    - file_path (str): The directory where the file will be saved.
    - file_name (str): The name of the file (without extension).
    
    Output:
    - None: The function saves the object to a .pt file at the specified location.
    """
    
    # Create the directory if it does not exist
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    # Full path to the file
    full_file_path = os.path.join(file_path, f"{file_name}.pt")
    
    # Delete the file if it already exists
    if os.path.exists(full_file_path):
        os.remove(full_file_path)
    
    # Save the object (which could be a tensor, model, or list of models) to the file
    torch.save(obj, full_file_path)


def extract_k_hop_subgraph(node_idx, num_hops, edge_index, num_nodes, x, y):
    """
    Extracts the k-hop subgraph of a node, including node features and labels.
    
    Parameters:
    - node_idx (int): The index of the central node.
    - num_hops (int): The number of hops to consider for the neighborhood.
    - edge_index (Tensor): The edge index tensor of the whole graph.
    - num_nodes (int): The total number of nodes in the whole graph.
    - x (Tensor): The node feature matrix of the whole graph.
    - y (Tensor): The node labels of the whole graph.
    
    Returns:
    - sub_data (Data): A PyG Data object representing the extracted subgraph, including node features and labels.
    """
    # Extract the k-hop subgraph around the specified node
    sub_nodes, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=node_idx,
        num_hops=num_hops,
        edge_index=edge_index,
        relabel_nodes=True,
        num_nodes=num_nodes,
    )
    
    # Extract features and labels for nodes in the subgraph
    sub_x = x[sub_nodes]
    sub_y = y[node_idx].view(-1,)
    
    # Create a subgraph data object including features and labels
    sub_data = Data(x=sub_x, edge_index=sub_edge_index, y=sub_y)
    return sub_data



def total_variation_distance(v,p=-1.):
    """
    Calculate the Total Variation distance between a given probability distribution tensor v
    and the uniform distribution.

    Parameters:
    - v (torch.Tensor): A 1D tensor of shape (N,) representing a probability distribution.

    Returns:
    - float: The Total Variation distance.
    """
    # Number of elements in v
    N = v.shape[0]

    uniform = torch.full_like(v, 1/N)
    # Calculate the Total Variation distance
    tv_distance = 0.5 * torch.sum(torch.abs(v - uniform))

    return tv_distance



class FeatureSelect(BaseTransform):
    def __init__(self, nfeats):
        """
        Initialize the transformation with the number of features to retain.

        Parameters:
        - feats (int): The number of features to retain from the beginning of the feature matrix.
        """
        self.nfeats = nfeats

    def __call__(self, data):
        """
        Retain only the first 'feats' features of the node feature matrix 'data.x'.
        Parameters:
        - data (torch_geometric.data.Data): The graph data object.
        Returns:
        - torch_geometric.data.Data: The modified graph data object with the node feature matrix sliced.
        """
        
        # Check if 'data.x' exists and has enough features
        data.x = data.x[:, :self.nfeats]
        return data



def size_split_idx(dataset, mode='ls'):

    num_graphs = len(dataset)
    num_val   = int(0.1 * num_graphs)
    num_test  = int(0.1 * num_graphs)
    num_train = num_graphs - num_test - num_val

    num_node_list = []
    train_idx = []
    valtest_list = []

    for data in dataset:
        num_node_list.append(data.num_nodes)

    sort_list = np.argsort(num_node_list)

    if mode == 'ls':
        train_idx = sort_list[2 * num_val:]
        valid_test_idx = sort_list[:2 * num_val]
    else:
        train_idx = sort_list[:-2 * num_val]
        valid_test_idx = sort_list[-2 * num_val:]
    random.shuffle(valid_test_idx)
    valid_idx = valid_test_idx[:num_val]
    test_idx = valid_test_idx[num_val:]

    split_idx = {'train': torch.tensor(train_idx, dtype = torch.long), 
                 'valid': torch.tensor(valid_idx, dtype = torch.long), 
                 'test': torch.tensor(test_idx, dtype = torch.long)}
    return split_idx



def cmd(x1, x2, n_moments=5):
    """
    central moment discrepancy (cmd)
    
    - Zellinger, Werner et al. "Robust unsupervised domain adaptation
    for neural networks via moment alignment," arXiv preprint arXiv:1711.06114,
    2017.
    """
    if isinstance(x1, torch.Tensor):
        x1 = x1.detach().cpu().numpy()
    if isinstance(x2, torch.Tensor):
        x2 = x2.detach().cpu().numpy()
    
    mx1 = x1.mean(0)
    mx2 = x2.mean(0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = l2diff(mx1,mx2)
    scms = dm
    for i in range(n_moments-1):
        # moment diff of centralized samples
        scms+=moment_diff(sx1,sx2,i+2)
    return scms


def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    val =  ((x1-x2)**2).sum()
    return np.sqrt(val)


def moment_diff(sx1, sx2, k):
    """
    Difference between the k-th moments of sx1 and sx2.
    """
    # Calculate the k-th moment for each set of samples
    ss1 = np.mean(sx1**k, axis=0)
    ss2 = np.mean(sx2**k, axis=0)
    
    # Calculate the L2 norm (Euclidean distance) between the two moments
    return l2diff(ss1, ss2)



def calc_pairwise_cmd(tensors, n_moments=5):
    """
    Calculate pairwise central moment discrepancy (CMD) between a list of tensors.
    
    Args:
    tensors (list of torch.Tensor): A list of tensors, where each tensor is of shape (N, D), 
                                    with N samples and D dimensions.
    n_moments (int): Number of moments to calculate. Default is 5.
    
    Returns:
    list: A list containing CMD values for each unique pair of tensors.
    """
    pairwise_cmds = []
    num_tensors = len(tensors)

    for i in range(num_tensors):
        for j in range(i + 1, num_tensors):  # To avoid recalculating 2-1 after 1-2
            
            cmd_value = cmd(tensors[i], tensors[j], n_moments)
            pairwise_cmds.append(cmd_value)
    return pairwise_cmds



def calc_nmi_clustering(tensors, env_labels, num_clusters=3,plot=False,fpath=None):
    """
    Perform MiniBatchKMeans clustering on each tensor and calculate NMI with env_labels.
    
    Args:
    tensors (list of torch.Tensor): A list of tensors, where each tensor is of shape (N, D).
    env_labels (torch.Tensor): A tensor of shape (N,), containing environment labels.
    num_clusters (int): The number of clusters to use in KMeans.
    
    Returns:
    list: A list containing NMI values for each tensor.
    """
    nmi_scores = []
    env_labels_np = env_labels.cpu().numpy()  # Convert env_labels to numpy if it's a tensor
    
    for i, tensor in enumerate(tensors):
        # Convert tensor to numpy for compatibility with sklearn
        tensor_np = tensor.cpu().numpy()

        # Perform MiniBatchKMeans clustering
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=1)
        predicted_labels = kmeans.fit_predict(tensor_np)

        # Calculate NMI between predicted clusters and environment labels
        nmi_score = normalized_mutual_info_score(env_labels_np, predicted_labels)
        nmi_scores.append(nmi_score)
        print (f"nmi scores:{nmi_score}")
        
        if plot:
            # Reduce the dimensionality to 2D using t-SNE
            tsne = TSNE(n_components=2, random_state=1)
            tensor_2d = tsne.fit_transform(tensor_np)

            # Generate filename for each plot
            filename = f'{fpath}/embedding_plot_epoch_{i+1}.pdf'

            # Call the plotting function with the 2D embedding and labels
            plot_embedding_with_labels(tensor_2d, env_labels_np, filename=filename)

    return nmi_scores


def plot_embedding_with_labels(embedding, labels, filename='embedding_plot.pdf'):
    """
    Plot a 2D embedding array with different colors for each label and save it to a PDF file.

    Parameters:
    embedding (numpy.ndarray): The 2D array of data points with shape (N, 2).
    labels (numpy.ndarray): The array of labels for each data point with shape (N,).
                             Labels are assumed to be in the set {0, 1, 2}.
    filename (str): The name of the PDF file to save the plot.

    Returns:
    None
    """
    # Define the colormap for the labels
    cmap = plt.get_cmap('Spectral')
    colors = cmap(np.linspace(0, 1, len(np.unique(labels))))
    
    # Create a scatter plot
    plt.figure()
    for label in np.unique(labels):
        label_mask = (labels == label)
        plt.scatter(embedding[label_mask, 0], embedding[label_mask, 1], 
                    color=colors[label], label=f'env label: {label}', s=30, edgecolor='k')
    
    plt.legend()
    
    # Save the plot to a PDF file
    
    plt.savefig(filename)
    plt.close()
    


def plot_cmd_dist(values, path):
    # Plotting histogram and KDE (linear scale)
    plt.figure(figsize=(15, 12))
    plt.hist(values, bins=60, density=True, color='blue', alpha=0.9, label='Histogram')

    # Plot KDE with seaborn (linear scale)
    sns.kdeplot(values, color='red', label='KDE', bw_adjust=0.5, linewidth=3)

    # Adding labels and title (linear scale)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel('Central Moment Distance', fontsize=32)
    plt.ylabel('Density', fontsize=32)
    plt.tight_layout()

    # Save the linear scale plot
    plt.savefig(path + 'cmd_dist.pdf')
    plt.close()

    # Plotting histogram and KDE (log scale)
    plt.figure(figsize=(15, 12))
    plt.hist(values, bins=60, density=True, color='blue', alpha=0.9, label='Histogram')

    # Plot KDE with seaborn (log scale)
    sns.kdeplot(values, color='red', label='KDE', bw_adjust=0.5, linewidth=3)

    # Set x-axis to log scale
    plt.xscale('log')

    # Adding labels and title (log scale)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel('Central Moment Distance (Log Scale)', fontsize=32)
    plt.ylabel('Density', fontsize=32)
    plt.tight_layout()

    # Save the log scale plot
    plt.savefig(path + 'cmd_dist_logscale.pdf')
    plt.close()



def load_model_and_calc_l2_dist(model, file_path):
    """
    Given a .pt file containing a list of model parameters and an initialized model, this function:
    1. Loads the saved state_dicts (model parameters) from the file.
    2. Loads each state_dict into the given model.
    3. Computes the L2 distance between the state_dict at the k^th and (k+1)^th positions.
    4. Returns a list of models and a list of corresponding L2 distances.

    Args:
        model (torch.nn.Module): Initialized PyTorch model.
        file_path (str): Path to the .pt file containing saved state_dicts.

    Returns:
        models (list): List of models with the loaded state_dicts.
        l2_distances (list): List of L2 distances between consecutive state_dicts.
    """
    
    # Load the list of state_dicts from the .pt file
    state_dicts = torch.load(file_path, map_location=torch.device('cpu'))
    
    # Initialize lists to store models and distances
    models = []
    l2_distances = []
    
    # Iterate over state_dicts to load them into the model and calculate L2 distances
    for i in range(len(state_dicts)):
        # Load the i-th state_dict into the model
        model.load_state_dict(state_dicts[i])
        # Set all parameters of the model to not require gradients
        for param in model.parameters():
            param.requires_grad = False
        
        models.append(deepcopy(model))  # Append the updated model to the list
        
        # If we have at least two state_dicts, calculate L2 distance between consecutive state_dicts
        if i > 0:
            prev_state_dict = state_dicts[i - 1]
            curr_state_dict = state_dicts[i]
            
            # Calculate L2 distance between the two state_dicts
            l2_dist = 0.0
            for key in curr_state_dict:
                l2_dist += torch.norm(curr_state_dict[key].float() - prev_state_dict[key].float(), p=2).item()
            
            l2_distances.append(l2_dist)
    
    return models, l2_distances



# Borrowed from https://github.com/fanyun-sun/InfoGraph
def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.
    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = np.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples

    if average:
        return Ep.mean()
    else:
        return Ep


# Borrowed from https://github.com/fanyun-sun/InfoGraph
def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.
    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = np.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples

    if average:
        return Eq.mean()
    else:
        return Eq
