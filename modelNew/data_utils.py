import os
import numpy as np
import pandas as pd
import re

import torch
from torch_geometric.data import Dataset,Data
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def split_dataset(dataset, p, num_splits=10, fpath='train_splits.npy'):
    """
    Randomly draws p% of samples for each label and performs ten such splits. 
    If the splits file already exists, it loads the saved splits instead of creating new ones.

    Args:
    - dataset (torch_geometric.data.Dataset): PyG dataset.
    - p (float): Percentage of samples to draw from each label (0 <= p <= 100).
    - num_splits (int): Number of splits to perform (default: 10).
    - fpath (str): Path to save or load the numpy array with split indices (default: 'train_splits.npy').

    Returns:
    - sub_train_datasets (list): List of subsets of the dataset with p% samples drawn for each label.
    """
    
    labels = dataset.data.y  # Assuming dataset has labels stored in `y`
    unique_labels = torch.unique(labels)
    sub_train_datasets = []

    if os.path.exists(fpath):
        # Load the splits if the file already exists
        print(f"Loading splits from {fpath}")
        all_splits = np.load(fpath, allow_pickle=True)
        
        # Create sub-train datasets from the loaded splits
        for split_indices in all_splits:
            split_indices = np.array(split_indices, dtype=np.int64)  # Ensure correct dtype
            sub_train_dataset = dataset[split_indices]
            sub_train_datasets.append(sub_train_dataset)

    else:
        all_splits = []

        for split_num in range(num_splits):
            split_indices = []

            for label in unique_labels:
                # Get indices of all samples with the current label
                label_indices = (labels == label).nonzero(as_tuple=True)[0]
                num_label_samples = len(label_indices)

                # Randomly select p% of the samples from the current label
                select_num = int(p * num_label_samples / 100)
                if select_num == 0:
                    continue
                selected_indices = torch.randperm(num_label_samples)[:select_num]
                split_indices.extend(label_indices[selected_indices].tolist())

            # Shuffle and store the split indices
            split_indices = np.array(split_indices)
            np.random.shuffle(split_indices)
            all_splits.append(split_indices)

            # Create sub-train dataset
            sub_train_dataset = dataset[split_indices]
            sub_train_datasets.append(sub_train_dataset)

        # Save the split indices to a numpy file
        np.save(fpath, np.array(all_splits, dtype=object))

    return sub_train_datasets



def parse_dir_name(dir_name):
    """
    Parse the directory name to extract parameters and their values.
    
    Args:
    - dir_name (str): The directory name to parse.
    
    Returns:
    - dict: A dictionary where keys are parameter names and values are parameter values.
    """
    # Split the directory name by '_' and parse
    parts = dir_name.split('_')
    params = {}
    for i in range(0, len(parts), 2):
        key = parts[i]
        try:
            value = parts[i + 1]
            # Attempt to convert numeric values
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            value = value  # Keep as string if conversion fails
        except IndexError:
            continue  # Skip if there's no value for a key
        params[key] = value
    return params

def load_dataframes(root_dir):
    """
    Load numpy arrays from subdirectories and return a pandas DataFrame.
    
    Args:
    - root_dir (str): The root directory to search recursively.
    
    Returns:
    - pd.DataFrame: A DataFrame where each row corresponds to a subdirectory's attributes and loaded numpy array.
    """
    data = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".npy"):
                # Load the numpy array
                npy_path = os.path.join(subdir, file)
                npy_array = np.load(npy_path)
                
                # Parse the directory name to get parameters
                dir_name = os.path.relpath(subdir, root_dir)
                params = parse_dir_name(dir_name.replace('/', '_'))
                
                # Add the numpy array to params dict
                params['numpyArray'] = npy_array
                
                # Append to data list
                data.append(params)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    return df


class DatasetWithClusterId(Dataset):
    def __init__(self, train_loader, train_dataset, model, n_clusters=10, device=0):
        """
        Initializes the DatasetWithClusterId class.

        Parameters:
        - train_loader: DataLoader for the training set.
        - train_dataset: Original dataset.
        - model: GNN model with an inference_encoding method.
        - n_clusters: Number of clusters for k-means clustering.
        - device: Device to run the model on.
        """
        super(DatasetWithClusterId, self).__init__(None, transform=None, pre_transform=None)
        self.train_loader = train_loader
        self.train_dataset = train_dataset
        self.data = train_dataset.data
        self.model = model
        self.model.to(device)
        self.n_clusters = n_clusters
        self.cluster_ids = None
        self.device = device
        print('DatasetWithClusterId initialized and clustering')
        self.process_cluster()

    def len(self):
        """
        Returns the length of the dataset.
        """
        return len(self.train_dataset)

    def get(self, idx):
        """
        Retrieves the data point at the given index and attaches the corresponding cluster ID and soft labels.

        Parameters:
        - idx: Index of the data point to retrieve.
        
        Returns:
        - data: Data object with added `cluster_id` and `cluster_label` attributes.
        """
        data = self.train_dataset[idx]
        data.cluster_id = torch.tensor([self.cluster_ids[idx]], dtype=torch.long)

        data.cluster_label = self.cluster_probs[idx].view(1,-1)
        return data

    def process_cluster(self):
        """
        Processes the dataset by inferring embeddings using the GNN model,
        applying k-means clustering, and training a calibrated SVM to get soft labels.
        """
        
        # Set model to evaluation mode
        self.model.eval()

        # Store all embeddings in a list
        all_embeddings = []

        # Infer embeddings for the entire dataset
        with torch.no_grad():
            for batch in self.train_loader:
                batch = batch.to(self.device)
                # Assuming batch has the required attributes for inference
                emb = self.model.inference_encoding(
                    batch.x, 
                    batch.edge_index, 
                    edge_attr=batch.edge_attr if 'edge_attr' in batch else None, 
                    edge_weight=None, 
                    batch=batch.batch if 'batch' in batch else None,
                    random_sampling=True,
                    node_cls=False
                )
                all_embeddings.append(emb.cpu().numpy())  # Convert to numpy

        # Concatenate embeddings into one array
        all_embeddings = np.concatenate(all_embeddings, axis=0)

        # Run minibatch k-means to get initial cluster IDs
        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters)
        initial_cluster_ids = kmeans.fit_predict(all_embeddings)
        self.cluster_ids = torch.tensor(initial_cluster_ids, dtype=torch.long)

        # Train LinearSVC and calibrate probabilities
        svc = LinearSVC()
        calibrated_svc = CalibratedClassifierCV(svc, cv=3)
        calibrated_svc.fit(all_embeddings, initial_cluster_ids)
        
        # Get probability estimates for each sample
        self.cluster_probs = calibrated_svc.predict_proba(all_embeddings)
        self.cluster_probs = torch.tensor(self.cluster_probs, dtype=torch.float)
        # self.cluster_probs shape: (num_samples, n_clusters)


def get_node_label(data: Data) -> torch.Tensor:
    """
    Given a PyTorch Geometric Data object, this function computes a boolean tensor (node_label)
    where each entry corresponds to whether the node is involved in an edge with the 
    'True' label in edge_gt.
    
    Args:
    - data (Data): A PyTorch Geometric Data object with attributes:
        - edge_index: Tensor of shape [2, num_edges] representing edge connections.
        - x: Tensor of node features with shape [num_nodes, feature_dim].
        - edge_gt: Boolean tensor of shape [num_edges], indicating the ground-truth label for each edge.
    
    Returns:
    - node_label (torch.Tensor): Boolean tensor of shape [num_nodes], where the value is True
      if the node is connected to an edge with a 'True' label in edge_gt, and False otherwise.
    """
    # Get the edge indices and the ground-truth labels for edges
    edge_index = data.edge_index  # Shape: [2, num_edges]
    edge_gt = data.edge_gt        # Shape: [num_edges]
    num_nodes = data.num_nodes    # Total number of nodes
    
    # Get the node indices corresponding to edges with edge_gt == True
    true_edges = edge_index[:, edge_gt]  # Select edges where edge_gt is True
    
    # Get unique node indices involved in these edges
    involved_nodes = torch.unique(true_edges)  # Shape: [num_involved_nodes]
    
    # Create a boolean tensor for node labels
    node_label = torch.zeros(num_nodes, dtype=torch.bool)  # Initialize with all False
    node_label[involved_nodes] = True  # Set True for involved nodes
    
    return node_label




# Example usage
if __name__ == "__main__":
    root_dir = "experiments/"
    df = load_dataframes(root_dir)
    print(df)


