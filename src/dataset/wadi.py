import os
import logging
import random
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as GraphData
from torch_geometric.utils import dense_to_sparse

from src.config import cfg


class CustomizedGraphData(GraphData):
    """https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html
    """
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.x.size(0)
        if key == 'edge_index_s': # structural edge index
            return self.x.size(0)
        return super().__inc__(key, value, *args, **kwargs)
    

SCALER_DICT = {
    'minmax': MinMaxScaler,
    'standard': StandardScaler
}


def get_spaced_randoms(n, start, end, m):
    """
    Generate n unique random numbers between start and end, 
    where each number is at least m steps away from the others.

    Args:
    n (int): The number of random numbers to generate.
    start (int): The start of the range in which to generate numbers.
    end (int): The end of the range in which to generate numbers.
    m (int): The minimum distance between any two numbers.

    Returns:
    list of int: The list of generated random numbers.
    """
    possible_nums = list(range(start, end + 1, m))
    if len(possible_nums) < n:
        raise ValueError(f"Cannot generate {n} numbers in the range {start}-{end} that are at least {m} apart.")
    random.seed(0)
    random_nums = random.sample(possible_nums, n)
    logging.info(f"[random_nums[0] : {random_nums}] Generated {n} random numbers in the range {start}-{end} that are at least {m} apart.")
    random_nums.sort()

    return random_nums


def split_indices(num_timestamps_train, test_size, ws, test_cont_ws=100):
    """
    Split the indices of the training set into train and test sets so that none of the test indices are within ws of train.
    """
    all_indices = set(range(num_timestamps_train))

    # Calculate the number of test samples
    num_test_samples = int(num_timestamps_train * test_size)

    test_start_indices = get_spaced_randoms(num_test_samples//test_cont_ws, 0, num_timestamps_train-test_cont_ws, ws)
    test_indices = []
    for i in test_start_indices:
        test_indices.extend(list(range(i, i+test_cont_ws)))

    train_indices = list(all_indices - set(test_indices))

    return sorted(train_indices), sorted(list(test_indices))


def get_discountinous_sequences(
        data, 
        label, 
        fault_labels=None, 
        sample_nums=None,
        win_size=15, 
        horizon=1, 
        stride=1
    ):
    n = data.shape[0]
    
    X_seq = []
    Y_seq = []
    Label_seq = []
    Fault_labels = []

    i = 0
    while i <= (n - win_size - horizon):
        # Check if all labels in the window are the same
        seq_has_same_label = len(set(label[i:i+win_size+horizon])) == 1
        if fault_labels is not None:
            seq_has_same_label = len(set(fault_labels[i:i+win_size+horizon])) == 1
            
        # Check if all sample numbers in the window are the same, if sample_nums is provided
        seq_has_same_sample = True  # Default to True
        if sample_nums is not None:
            seq_has_same_sample = len(set(sample_nums[i:i+win_size+horizon])) == 1
        
        if seq_has_same_label and seq_has_same_sample:
            X_seq.append(data[i:i+win_size])
            Y_seq.append(data[i+win_size:i+win_size+horizon])
            Label_seq.append(label[i])
            if fault_labels is not None:
                Fault_labels.append(fault_labels[i])
            i += stride
        else:
            # Skip ahead to the next non-overlapping window
            i += win_size

    Fault_labels = np.array(Fault_labels) if fault_labels is not None else None
    return np.array(X_seq), np.array(Y_seq), np.array(Label_seq), Fault_labels


def get_sequences(data, label, fault_labels=None, win_size=32, horizon=1):
    # k, n, nf
    n_samples = data.shape[0]
    
    x_offsets = np.arange(-win_size+1, 1)
    y_offsets = np.arange(1, 1+horizon)
    
    xs, ys, ts, fl = [], [], [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(n_samples - abs(max(y_offsets)))  # Exclusive    
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        label_t = max(label[t + x_offsets]) # if any then marked as abnormal
        if fault_labels is not None:
            fault_label_t = max(fault_labels[t + x_offsets]) # if any then marked as abnormal
            fl.append(fault_label_t)
        xs.append(x_t)
        ys.append(y_t)
        ts.append(label_t)
    X = np.stack(xs, axis=0)
    Y = np.stack(ys, axis=0)
    Label = np.stack(ts, axis=0)
    if fault_labels is not None:
        Fault = np.stack(fl, axis=0)
        return X, Y, Label, Fault
    return X, Y, Label, None


def get_graph_sequences(
    X, Y, 
    edge_index, 
    labels, 
    edge_attr=None,
    edge_index_s=None,
    aug_convar_on_node=False, 
    var_start_idx=2,
    map_convar=False, 
    faults=None
):
    """Prepare list of sliding window data in graph format 
    Args:
    """
    X = X.astype(np.float32) # (n_samples, seq_len, n_nodes)
    Y = Y.astype(np.float32) # (n_samples, 1, n_nodes)
    labels = labels.astype(np.int32).reshape(-1, 1) # (n_samples, 1)
    if faults is not None:
        faults = faults.astype(np.int32).reshape(-1, 1) # (n_samples, 1)
    
    # n_nodes = X.shape[2]
    
    graph_list = []
    for i in range(len(X)):
        xi = torch.from_numpy(X[i].T) # [n_nodes, seq_len]
        yi = torch.from_numpy(Y[i].T) # [n_nodes, seq_len]
        label = torch.from_numpy(labels[i])
        args = {
            'x': xi,
            'y': yi,
            'edge_index': edge_index,
            'label': label
        }
        if edge_index_s is not None:
            args['edge_index_s'] = edge_index_s
        if edge_attr is not None:
            args['edge_attr'] = edge_attr
        # TODO: currenlty the index are fixed so the control vars are at the beginning
        if aug_convar_on_node:
            ci = xi[:var_start_idx, :]
            xi = xi[var_start_idx:, :]
            yi = yi[var_start_idx:, :]
            args['x'] = xi
            args['c'] = ci
            args['y'] = yi
        elif map_convar:
            ci = xi[:var_start_idx, :]
            yi = xi[var_start_idx:, :]
            args['x'] = ci
            args['y'] = yi
        if faults is not None:
            args['fault'] = torch.from_numpy(faults[i])
        g = CustomizedGraphData(**args) if edge_index_s is not None else GraphData(**args)
        graph_list.append(g)
    return graph_list


def load_data(
    dataset_dir, 
    win_size=32,
    horizon=1,
    aug_convar_on_node=False,
    map_convar=False,
    return_scaler=False,
):
    test_fault_labels = None
    normalize = cfg.dataset.normalize
    
    CONV_VARS = []
    VARS = []
    
    # TODO: fix run path
    dataset_dir = f'run/{dataset_dir}'
    sample_nums_train = None
    sample_nums_test = None

    if 'tep_rieth2017' in dataset_dir:
        normalize = True
        VARS = [f'xmeas_{i+1}' for i in range(22)] + [f'xmv_{i+1}' for i in range(11)]
        train_file_path = f'{dataset_dir}/tep_train.csv'
        df_train_normal = pd.read_csv(train_file_path, index_col=0)
    
        test_file_path = f'{dataset_dir}/tep_test.csv'
        df_test_faulty = pd.read_csv(test_file_path, index_col=0)
    
        sample_len = df_train_normal['sample'].max()
        mask = df_train_normal['sample'] <= sample_len - sample_len%win_size
        X_train = np.array(df_train_normal[mask][VARS].values).astype(np.float32)
        # X_train = np.array(df_train_normal[mask].iloc[:, 3:].values).astype(np.float32)
        train_labels = np.zeros(X_train.shape[0])
    
        sample_len = df_test_faulty['sample'].max()
        mask = df_test_faulty['sample'] <= sample_len - sample_len%win_size
        X_test = np.array(df_test_faulty[mask][VARS].values).astype(np.float32)
        
        test_labels = np.array(df_test_faulty[mask].iloc[:, -1]).astype(np.int32)
        test_fault_labels = np.array(df_test_faulty[mask].iloc[:, 0]).astype(np.int32)

    elif 'wadi' in dataset_dir:
        normalize = False
        train_file_path = f'{dataset_dir}/{cfg.dataset.train_file}'
        test_file_path = f'{dataset_dir}/{cfg.dataset.test_file}'
        df_train_normal =  pd.read_csv(train_file_path, index_col=0)
        df_test =  pd.read_csv(test_file_path, index_col=0)
        # # read convars from txt file
        # convar_file_path = f'{dataset_dir}/convars.txt'
        # with open(convar_file_path, 'r') as file:
        #     CONV_VARS = file.read().splitlines()
            
        VARS = [c for c in df_train_normal.columns if c not in ['attack']]
        CONV_VARS = []

        # if cfg.dataset.no_convar:
        #     CONV_VARS = []

        # TODO fix this order thing!!!!!!!
        # everything except for the last column
        X_train = np.array(df_train_normal.values).astype(np.float32)[:, :-1]
        X_test = np.array(df_test.values).astype(np.float32)[:, :-1]
        
        train_labels = np.array(df_train_normal['attack']).astype(bool)
        test_labels = np.array(df_test['attack']).astype(bool)

    elif 'contr/robot5' in dataset_dir:
        CONV_VARS = [f'u_{i+1}' for i in range(5)]
        VARS = [f'y_{i+1}' for i in range(10)]
        train_file_path = f'{dataset_dir}/{cfg.dataset.train_file}'
        test_file_path = f'{dataset_dir}/{cfg.dataset.test_file}'
        df_train_normal =  pd.read_csv(train_file_path, index_col=0)
        df_test =  pd.read_csv(test_file_path, index_col=0)
        X_train = np.array(df_train_normal.values).astype(np.float32)
        X_test = np.array(df_test.values).astype(np.float32)
        train_labels = X_train[:, -1]
        test_labels = X_test[:, -1]
        X_train = X_train[:, :-1]
        X_test = X_test[:, :-1]

    elif 'synthetic' in dataset_dir:
        if 'syn_5' in dataset_dir or 'syn_6' in dataset_dir:
            CONV_VARS = [f'u_{i+1}' for i in range(4)]
            VARS = [f'y_{i+1}' for i in range(10)]
        else:
            CONV_VARS = [f'x_{i+1}' for i in range(2)]
            VARS = [f'y_{i+1}' for i in range(5)]
        train_file_path = f'{dataset_dir}/{cfg.dataset.train_file}'
        test_file_path = f'{dataset_dir}/{cfg.dataset.test_file}'
        df_train_normal =  pd.read_csv(train_file_path, index_col=0)
        df_test =  pd.read_csv(test_file_path, index_col=0)
        X_train = np.array(df_train_normal.values).astype(np.float32)
        X_test = np.array(df_test.values).astype(np.float32)
        train_labels = X_train[:, -1]
        test_labels = X_test[:, -1]
        X_train = X_train[:, :-1]
        X_test = X_test[:, :-1]
        
    elif 'pronto' in dataset_dir:
        train_file_path = f'{dataset_dir}/{cfg.dataset.train_file}'
        df_train_normal =  pd.read_csv(train_file_path)

        columns = df_train_normal.columns
        if cfg.dataset.aug_convar:
            if cfg.dataset.use_indep_vars:
                CONV_VARS = ['AirIn', 'WaterIn', 'Air.T', 'Water.T']
            else:
                CONV_VARS = ['AirIn', 'WaterIn']
        else:
            CONV_VARS = []
        # VARS is the rest of the columns
        VARS = [c for c in columns if c not in CONV_VARS+['day', 'OC', 'Fault']]
        if cfg.dataset.no_convar:
            CONV_VARS = []

        test_file_path = f'{dataset_dir}/{cfg.dataset.test_file}'
        df_test =  pd.read_csv(test_file_path)
        
        X_train = np.array(df_train_normal[CONV_VARS+VARS].values).astype(np.float32)
        logging.info("Feature order %s", CONV_VARS + VARS)
        X_test = np.array(df_test[CONV_VARS+VARS].values).astype(np.float32)
        train_labels =  np.array(df_train_normal['Fault']!='Normal').astype(np.int32)
        test_labels = np.array(df_test['Fault']!='Normal').astype(np.int32)
        # Define a dictionary to map fault names to integers
        fault_map = {'Normal': 0, 'Air blockage': 1, 'Air leakage': 2, 'Diverted flow': 3, 'Slugging': 4}
        cfg.dataset.num_classes = df_test['Fault'].nunique()
        df_test['Fault'] = df_test['Fault'].map(fault_map)
        test_fault_labels = np.array(df_test['Fault']).astype(np.int32)
    
    elif 'metallicadour' in dataset_dir:
        train_file_path = f'{dataset_dir}/{cfg.dataset.train_file}'
        df_train_normal =  pd.read_csv(train_file_path)

        columns = df_train_normal.columns
        CONV_VARS = ['current_x', 'current_y', 'current_z']
        # VARS is the rest of the columns
        VARS = ['force_x', 'force_y', 'force_z', 'torque_x', 'torque_y', 'torque_z']
        test_file_path = f'{dataset_dir}/{cfg.dataset.test_file}'
        df_test =  pd.read_csv(test_file_path)
        
        X_train = np.array(df_train_normal[CONV_VARS+VARS].values).astype(np.float32)
        X_test = np.array(df_test[CONV_VARS+VARS].values).astype(np.float32)
        train_labels =  np.array(df_train_normal['tool_state']!='healthy').astype(np.int32)
        test_labels = np.array(df_test['tool_state']!='healthy').astype(np.int32)
        # Define a dictionary to map fault names to integers
        fault_map = {'healthy': 0, 'broken_tooth': 1, 'surface_damage': 2, 'flack_damage': 3}
        cfg.dataset.num_classes = df_test['tool_state'].nunique()
        df_test['tool_state'] = df_test['tool_state'].map(fault_map)
        test_fault_labels = np.array(df_test['tool_state']).astype(np.int32)
        sample_nums_train = np.array(df_train_normal['sample_num']).astype(np.int32)
        sample_nums_test = np.array(df_test['sample_num']).astype(np.int32)
    
    else:
        raise ValueError(f"dataset_dir {dataset_dir} not surported")
    
    if aug_convar_on_node:
        num_nodes = len(VARS)   
        cfg.dataset.convar_dim = len(CONV_VARS)
    elif map_convar:
        num_nodes = len(CONV_VARS)
    else:
        num_nodes = len(VARS) + len(CONV_VARS)
    
    if cfg.dataset.has_adj:
        adj_file = f'{dataset_dir}/adj.npy'
        edge_index_file = f'{dataset_dir}/edge_index.npy'
        # check if adj_file available
        if os.path.exists(adj_file):
            # graph structure
            init_adj = torch.from_numpy(np.load(adj_file))
            edge_index, edge_feat = dense_to_sparse(init_adj)
            logging.info(f"Loaded graph structure from {adj_file}, graph has {edge_index.shape[1]} edges")
        elif os.path.exists(edge_index_file):
            edge_index = torch.from_numpy(np.load(edge_index_file))
            edge_index = torch.sub(edge_index, len(CONV_VARS))
            edge_feat = torch.ones(edge_index.shape[1])
            logging.info(f"Loaded graph structure from {edge_index_file}, graph has {edge_index.shape[1]} edges")
        else:
            raise ValueError(f"cfg.dataset.has_adj set to true, but there is no 'adj.npy' or 'edge_index.npy' available in {dataset_dir}")
    else:
        init_adj = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
        # init_adj = np.ones((num_nodes, num_nodes))
        init_adj = init_adj.astype(np.float32)
        # graph structure
        init_adj = torch.from_numpy(init_adj)
        edge_index, edge_feat = dense_to_sparse(init_adj)
    
    edge_index_s = None
    if 'pronto' in dataset_dir and cfg.dataset.reg_by_prior_graph:
        edge_index_file = f'{dataset_dir}/edge_index.npy'
        edge_index_s = torch.from_numpy(np.load(edge_index_file))
        edge_index_s = torch.sub(edge_index_s, len(CONV_VARS))

    num_edges = len(edge_index[0])
    edge_feat = edge_feat.reshape(-1, 1)
    
    idx_start = 0
    num_feats = 1
    scaler = SCALER_DICT[cfg.dataset.scaler_type]()
    
    graph_datasets = []

    if 'pronto' in dataset_dir or 'metallicadour' in dataset_dir:
        # get sequence data
        X_seq, Y_seq, Label_seq, _ = get_discountinous_sequences(
            X_train, 
            train_labels, 
            sample_nums=sample_nums_train,
            win_size=win_size,
            horizon=horizon,
            stride=cfg.dataset.stride,
        )
        if 'v3' in cfg.case:
            cfg.dataset.test_split = 0.2
        train_indices, test_indices = split_indices(X_seq.shape[0], 
                                                    test_size=cfg.dataset.test_split, 
                                                    ws=win_size)
        
        X_train_seq = X_seq[train_indices]
        Y_train_seq = Y_seq[train_indices]
        X_test_seq = X_seq[test_indices]
        Y_test_seq = Y_seq[test_indices]
        Label_train_seq = Label_seq[train_indices]
        Label_test_seq = Label_seq[test_indices]
        
        # normalize data split
        if normalize:
            scaler.fit(X_train_seq[:, 0, :])

            # loop over time steps
            for i in range(X_train_seq.shape[1]):
                X_train_seq[:, i, :] = scaler.transform(X_train_seq[:, i, :])
            for i in range(Y_train_seq.shape[1]):
                Y_train_seq[:, i, :] = scaler.transform(Y_train_seq[:, i, :])
                
            for i in range(X_test_seq.shape[1]):
                X_test_seq[:, i, :] = scaler.transform(X_test_seq[:, i, :])
            for i in range(Y_test_seq.shape[1]):
                Y_test_seq[:, i, :] = scaler.transform(Y_test_seq[:, i, :])
        
        # get graph sequence data
        graph_list_train = get_graph_sequences(
            X=X_train_seq, 
            Y=Y_train_seq,
            edge_index=edge_index,
            edge_index_s=edge_index_s,
            labels=Label_train_seq,
            var_start_idx=len(CONV_VARS),
            map_convar=map_convar,
            aug_convar_on_node=aug_convar_on_node,
        )
        
        if 'v3' in cfg.case:
            cfg.dataset.val_split = 1/8
        graph_list_train, graph_list_eval = train_test_split(graph_list_train, shuffle=True, random_state=cfg.seed, test_size=cfg.dataset.val_split)
        
        graph_datasets.append(graph_list_train)
        graph_datasets.append(graph_list_eval)
        
        X_seq_f, Y_seq_f, Label_seq_f, Fault_seq_f = get_discountinous_sequences(
            X_test, 
            test_labels, 
            sample_nums=sample_nums_test,
            win_size=win_size,
            horizon=horizon,
            fault_labels=test_fault_labels,
            stride=cfg.dataset.stride,
        )
        
        if normalize:
            # loop over time steps
            for i in range(X_seq_f.shape[1]):
                X_seq_f[:, i, :] = scaler.transform(X_seq_f[:, i, :])
            for i in range(Y_seq_f.shape[1]):
                Y_seq_f[:, i, :] = scaler.transform(Y_seq_f[:, i, :])

        graph_list_test = get_graph_sequences(
            X=np.concatenate((X_test_seq, X_seq_f), axis=0), 
            Y=np.concatenate((Y_test_seq, Y_seq_f), axis=0),
            edge_index=edge_index,
            edge_attr=edge_feat,
            edge_index_s=edge_index_s,
            labels=np.concatenate((Label_test_seq, Label_seq_f), axis=0),
            faults=np.concatenate((Label_test_seq, Fault_seq_f), axis=0),
            var_start_idx=len(CONV_VARS),
            map_convar=map_convar,
            aug_convar_on_node=aug_convar_on_node,
        )

        graph_datasets.append(graph_list_test)
        test_labels = np.concatenate([Label_seq_f, Label_seq[test_indices]])
        
    else:
        # train and eval
        num_timestamps_train = X_train.shape[0]
        indices = np.arange(num_timestamps_train)
        # train val split
        for split_ratio in [0.8, 0.2]:
            idx_split = round(num_timestamps_train * split_ratio)
            idx_end = min(idx_start+idx_split, num_timestamps_train)
            
            # raw data split
            data = X_train[indices[idx_start:idx_end]]
            label = train_labels[indices[idx_start:idx_end]]
            if idx_start == 0:
                scaler.fit(data)

            # normalize data split
            if normalize:
                data = scaler.transform(data)
            
            # get sequence data
            X_seq, Y_seq, Label_seq, _ = get_sequences(
                data, label, 
                win_size=win_size,
                horizon=horizon
            )
            
            # get graph sequence data
            graph_list = get_graph_sequences(
                X=X_seq, 
                Y=Y_seq,
                edge_index=edge_index,
                edge_attr=edge_feat,
                labels=Label_seq,
                var_start_idx=len(CONV_VARS),
                map_convar=map_convar,
                aug_convar_on_node=aug_convar_on_node,
            )
            graph_datasets.append(graph_list)
            idx_start = idx_end

        # test
        # normalize test
        if normalize:
            X_test = scaler.transform(X_test)
            
        # get sequence data
        X_seq_test, Y_seq_test, Label_seq_test, Fault_seq_test = get_sequences(
            X_test, test_labels, 
            fault_labels=test_fault_labels, 
            win_size=win_size, horizon=horizon                                             
        )
    
        # get graph sequence data
        graph_list_test = get_graph_sequences(
            X=X_seq_test, 
            Y=Y_seq_test,
            edge_index=edge_index,
            labels=Label_seq_test,
            faults=Fault_seq_test,
            var_start_idx=len(CONV_VARS),
            aug_convar_on_node=aug_convar_on_node,
            map_convar=map_convar,
        )
        graph_datasets.append(graph_list_test)


    n_train_graphs = len(graph_datasets[0])
    n_val_graphs = len(graph_datasets[1])
    n_test_graphs = len(graph_datasets[2])
    num_graphs = n_train_graphs + n_val_graphs + n_test_graphs
    
    cfg.dataset.n_train = n_train_graphs
    cfg.dataset.n_val = n_val_graphs
    cfg.dataset.n_test = n_test_graphs
    
    cfg.dataset.n_nodes = num_nodes
    cfg.dataset.n_edges = num_edges
    
    d_info = '\n'
    d_info += f'Loaded data from {dataset_dir}\n'
    d_info += f'------------ Basic -----------\n'
    d_info += f'# Samples: {num_graphs}\n'
    d_info += f'# Node: {num_nodes}\n'
    d_info += f'# Edges: {num_edges}\n'
    d_info += f'# Sequence Length: {win_size}\n'
    
    d_info += f'------------ Anomaly rate -----------\n'
    d_info += f'# Anomaly rate train: {sum(train_labels)/len(train_labels):.3f}\n'
    d_info += f'# Anomaly rate test: {sum(test_labels)/len(test_labels):.3f}\n'
    
    d_info += f'------------ Feature -----------\n'
    d_info += f'Node feature size: {num_feats}\n'
    d_info += f'Edge feature size: 1\n'
    
    if aug_convar_on_node:
        d_info += f'------------ Control -----------\n'
        d_info += f'Control feature size: {cfg.dataset.convar_dim}\n'

    if normalize:
        d_info += f'------------ Scaling -----------\n'
        if cfg.dataset.scaler_type == 'minmax':
            d_info += f'{np.array2string(scaler.data_range_, precision=2,)}\n'
        elif cfg.dataset.scaler_type == 'standard':
            d_info += f'{np.array2string(scaler.mean_, precision=2,)}\n'
    
    d_info += f'------------ Split -----------\n'
    d_info += f'   Train: {n_train_graphs}/{num_graphs}\n'
    d_info += f'Validate: {n_val_graphs}/{num_graphs}\n'
    d_info += f'    Test: {n_test_graphs}/{num_graphs}\n'
    logging.info(d_info)
    
    if return_scaler:
        return graph_datasets, scaler
    else:
        return graph_datasets


def create_data_loaders(datasets, batch_size=16):
    d_train, d_val, d_test = datasets

    train_loader = DataLoader(d_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(d_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(d_test, batch_size=batch_size, shuffle=False)      
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    graph_datasets = load_data('datasets/tep_rieth2017')
    loaders = create_data_loaders(graph_datasets)