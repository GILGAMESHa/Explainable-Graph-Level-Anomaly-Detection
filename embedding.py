import numpy as np
import copy
from sklearn import preprocessing

def wl(adj, feat, h):
    degrees = np.sum(adj, axis=1)
    degrees[degrees == 0] = 1
    graph_feat = []
    feat = preprocessing.normalize(feat, norm='l2', axis=0)
    graph_feat.append(feat)
    phi = feat.copy()
    for _ in range(h):
        neighbor_sum = adj.dot(phi)
        phi = 0.5 * (phi + (1 / degrees[:, np.newaxis]) * neighbor_sum)
        graph_feat.append(phi)
    return np.concatenate(graph_feat, axis=1)

def WL_noconcate_fast(node_features, adj_mat):
    embedding = np.dot(adj_mat, node_features)
    return embedding

def create_adj_avg(adj_mat):
    np.fill_diagonal(adj_mat, 0)
    adj = copy.deepcopy(adj_mat)
    deg = np.sum(adj, axis=1)
    deg[deg == 0] = 1
    deg = (1/deg) * 0.5
    deg_mat = np.diag(deg)
    adj = deg_mat.dot(adj_mat)
    np.fill_diagonal(adj, 0.5)
    return adj

def createWlEmbedding_fast(adj, feat, h):
    new_adj = create_adj_avg(adj)
    embedding = preprocessing.normalize(feat, norm='l2', axis=0)
    for it in range(h+1):
        embedding = WL_noconcate_fast(embedding, new_adj)
    return embedding

def createWlEmbedding_1(adj, feat, h):
    graph_feat = []
    new_adj = create_adj_avg(adj)
    embedding = preprocessing.normalize(feat, norm='l2', axis=0)
    for it in range(h+1):
        embedding = WL_noconcate_fast(embedding, new_adj)
        graph_feat.append(embedding)
    return np.concatenate(graph_feat, axis=1)