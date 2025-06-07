from torch_geometric.datasets import TUDataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
import os
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder
from collections import Counter

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def balance_labels(label):
    unique_labels, counts = np.unique(label, return_counts=True)
    min_count_idx = np.argmin(counts)
    balanced_label = np.where(label == unique_labels[min_count_idx], 0, 1)
    return balanced_label

def one_hot_encode_features(feature_train, feature_test):
    # 合并训练集和测试集
    all_features = feature_train + feature_test
    # 将所有特征矩阵合并为一个数组
    all_labels = np.vstack(all_features)
    # 进行 one-hot 编码
    encoder = OneHotEncoder(sparse_output=False, dtype=int)
    all_encoded = encoder.fit_transform(all_labels)
    # 重新拆分回 feature_train 和 feature_test
    split_indices = [len(matrix) for matrix in feature_train]
    feature_train_encoded = np.split(all_encoded[:sum(split_indices)], np.cumsum(split_indices)[:-1])
    feature_test_encoded = np.split(all_encoded[sum(split_indices):],
                                    np.cumsum([len(matrix) for matrix in feature_test])[:-1])
    # 转换回列表格式
    feature_train_encoded = [matrix for matrix in feature_train_encoded]
    feature_test_encoded = [matrix for matrix in feature_test_encoded]
    return feature_train_encoded, feature_test_encoded

def Tudata_split(y, seed=1):
    kfd = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    indices = np.arange(len(y))
    splits = []
    for fold, (train_idx, test_idx) in enumerate(kfd.split(indices, y)):
        splits.append({
            'train_idx': np.array(train_idx, dtype=np.int64),
            'test_idx': np.array(test_idx, dtype=np.int64)
        })
    return splits

def load_data(ds_name, attr):
    _ = TUDataset(root="../data/TUDataset", name=ds_name, use_node_attr=True)
    path = f'../data/TUDataset/{ds_name}/raw/{ds_name}'
    graph_indicator = np.loadtxt(f"{path}_graph_indicator.txt", dtype=np.int64)
    _, graph_size = np.unique(graph_indicator, return_counts=True)

    edges = np.loadtxt(f"{path}_A.txt", dtype=np.int64, delimiter=",")
    edges -= 1
    A = csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(graph_indicator.size, graph_indicator.size))

    if attr == 'node attribute':
        node_attri_path = f"{path}_node_attributes.txt"
        if os.path.exists(node_attri_path):
            x = np.loadtxt(node_attri_path, delimiter=',', dtype=np.float64)
            mean = np.mean(x, axis=0, keepdims=True)
            std = np.std(x, axis=0, keepdims=True)
            x = (x - mean) / std
            print('Use node attribute as attr')
            # x = x / np.linalg.norm(x, axis=1, keepdims=True)
        else:
            print('node attribute file not found!')
    elif attr == 'node label':
        node_labels_path = f"{path}_node_labels.txt"
        if os.path.exists(node_labels_path):
            x = np.loadtxt(node_labels_path, dtype=np.int64).reshape(-1, 1)
            enc = OneHotEncoder(sparse_output=False)
            x = enc.fit_transform(x)
            print('Use node label as attr')
        else:
            print('node label file not found!')
    else:
        x = A.sum(axis=1)
        x = np.asarray(x)
        mean = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True)
        x = (x - mean) / std
        print('Use node degree as attr')
    adj = []
    features = []
    idx = 0
    for i in range(graph_size.size):
        adj.append(A[idx:idx + graph_size[i], idx:idx + graph_size[i]])
        features.append(x[idx:idx + graph_size[i], :])
        idx += graph_size[i]

    class_labels = np.loadtxt(f"{path}_graph_labels.txt", dtype=np.int64)
    class_labels = balance_labels(class_labels)
    unique_classes, counts = np.unique(class_labels, return_counts=True)
    # print('Raw class label:\n')
    # for cls, count in zip(unique_classes, counts):
    #     print(f"类别 {cls}: {count} 个样本")

    return adj, features, class_labels

def load_data_select(ds_name, attr, selected_class=None):
    _ = TUDataset(root="../data/TUDataset", name=ds_name, use_node_attr=True)
    path = f'../data/TUDataset/{ds_name}/raw/{ds_name}'
    graph_indicator = np.loadtxt(f"{path}_graph_indicator.txt", dtype=np.int64)
    _, graph_size = np.unique(graph_indicator, return_counts=True)

    edges = np.loadtxt(f"{path}_A.txt", dtype=np.int64, delimiter=",")
    edges -= 1
    A = csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(graph_indicator.size, graph_indicator.size))

    if attr == 'node attribute':
        node_attri_path = f"{path}_node_attributes.txt"
        if os.path.exists(node_attri_path):
            x = np.loadtxt(node_attri_path, delimiter=',', dtype=np.float64)
            mean = np.mean(x, axis=0, keepdims=True)
            std = np.std(x, axis=0, keepdims=True)
            x = (x - mean) / std
            print('Use node attribute as attr')
        else:
            print('node attribute file not found!')
    elif attr == 'node label':
        node_labels_path = f"{path}_node_labels.txt"
        if os.path.exists(node_labels_path):
            x = np.loadtxt(node_labels_path, dtype=np.int64).reshape(-1, 1)
            enc = OneHotEncoder(sparse_output=False)
            x = enc.fit_transform(x)
            print('Use node label as attr')
        else:
            print('node label file not found!')
    else:
        x = A.sum(axis=1)
        x = np.asarray(x)
        mean = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True)
        x = (x - mean) / std
        print('Use node degree as attr')

    adj = []
    features = []
    idx = 0
    for size in graph_size:
        adj.append(A[idx:idx + size, idx:idx + size])
        features.append(x[idx:idx + size, :])
        idx += size

    class_labels = np.loadtxt(f"{path}_graph_labels.txt", dtype=np.int64)
    class_labels = balance_labels(class_labels)

    if selected_class is not None:
        selected_class = int(selected_class)
        selected_indices = np.where(class_labels == selected_class)[0]
        adj = [adj[i] for i in selected_indices]
        features = [features[i] for i in selected_indices]
        class_labels = class_labels[selected_indices]
        print(f"只保留了类 {selected_class} 的数据，共 {len(selected_indices)} 个图")

    return adj, features, class_labels

def load_tox(ds_name, subtype, attr):
    TUDataset(root="../data/TUDataset", name=f'{ds_name}_{subtype}', use_node_attr=True)
    graph_indicator = np.loadtxt(f"../data/TUDataset/{ds_name}_{subtype}/raw/{ds_name}_{subtype}_graph_indicator.txt", dtype=np.int64)
    _, graph_size = np.unique(graph_indicator, return_counts=True)

    edges = np.loadtxt(f"../data/TUDataset/{ds_name}_{subtype}/raw/{ds_name}_{subtype}_A.txt", dtype=np.int64, delimiter=",")
    edges -= 1
    A = csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                   shape=(graph_indicator.size, graph_indicator.size))

    node_labels_path = f"../data/TUDataset/{ds_name}_{subtype}/raw/{ds_name}_{subtype}_node_labels.txt"

    if attr == 'node label':
        x = np.loadtxt(node_labels_path, dtype=np.int64).reshape(-1, 1)
    else:
        x = A.sum(axis=1)
    adj = []
    features = []
    idx = 0
    for i in range(graph_size.size):
        adj.append(A[idx:idx + graph_size[i], idx:idx + graph_size[i]])
        features.append(x[idx:idx + graph_size[i], :])
        idx += graph_size[i]

    class_labels = np.loadtxt(f"../data/TUDataset/{ds_name}_{subtype}/raw/{ds_name}_{subtype}_graph_labels.txt", dtype=np.int64)
    return adj, features, class_labels, graph_size

def load_data_tox(ds_name, attr):
    adj_train, x_train, class_labels_train, graph_size_train = load_tox(ds_name, 'training', attr)
    adj_test, x_test, class_labels_test, graph_size_test = load_tox(ds_name, 'testing', attr)
    if attr == 'node label':
        x_train, x_test = one_hot_encode_features(x_train, x_test)
    adj_train = convert2dense(adj_train)
    adj_test = convert2dense(adj_test)
    return adj_train, x_train, class_labels_train, adj_test, x_test, class_labels_test

def data_split(adj, features, class_labels, seed, fold):
    adj_list = convert2dense(adj)  # 邻接矩阵转换
    class_labels = balance_labels(class_labels)  # 处理类别
    splits = Tudata_split(class_labels, seed)  # 生成数据划分

    data_splits = {
        'train_adj': [],
        'test_adj': [],
        'train_x': [],
        'test_x': [],
        'train_class_labels': [],
        'test_class_labels': []
    }

    for f in range(fold):
        train_idx, test_idx = splits[f]['train_idx'], splits[f]['test_idx']

        data_splits['train_adj'].append([adj_list[i] for i in train_idx])
        data_splits['test_adj'].append([adj_list[j] for j in test_idx])
        data_splits['train_x'].append([features[i] for i in train_idx])
        data_splits['test_x'].append([features[j] for j in test_idx])
        data_splits['train_class_labels'].append(class_labels[train_idx])
        data_splits['test_class_labels'].append(class_labels[test_idx])

    return data_splits

def convert2dense(adj_list):
    dense_adj_list = []
    for adj in adj_list:
        dense_adj = adj.toarray()
        dense_adj_list.append(dense_adj)
    return dense_adj_list


if __name__ == '__main__':
    dataset = 'AIDS'
    # input_dim, train_loader, test_loader = build_loader(dataset, 64, 1, 1)
    # data_iter = iter(train_loader)
    # max_num_nodes = next(data_iter)['adj'].shape[1]
    # print(max_num_nodes)