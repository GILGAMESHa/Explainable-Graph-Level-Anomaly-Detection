from embedding import *
from utils.util import *
import argparse
from sklearn.metrics import roc_auc_score
import time
import random
from utils.IDK2 import *
from utils.load_data import *
from utils.visualize_embedding import *
from utils.vis_graph import *
from utils.ik_inne_gpu_v1 import *
from cluster import *

def find_most_similar_normal_samples_with_neighbors2(
        features,
        scores,
        anomaly_ratio=0.1,
        top_k_anomalies=5,
        top_k_neighbors=5):
    num_samples   = len(scores)
    num_anomalies = int(num_samples * anomaly_ratio)

    # 分数越低越异常 → 升序
    sorted_idx       = np.argsort(scores)
    anomaly_indices  = sorted_idx[:num_anomalies]
    normal_indices   = sorted_idx[num_anomalies:]

    top_anomalies = anomaly_indices[:top_k_anomalies]

    results = {}
    for anomaly in top_anomalies:
        a_feat = features[anomaly]

        # 1) 找与 anomaly 最相似的 normal
        sim_to_normals   = np.dot(features[normal_indices], a_feat)
        top_sim_normal   = normal_indices[np.argmax(sim_to_normals)]

        # 2) 找与 anomaly 最相似的 top-k 邻居（可包括 normal，也可不包括）
        sims_all         = np.dot(features[normal_indices], a_feat).astype(np.float64)
        #sims_all[anomaly] = -np.inf                     # 排除自身
        anomaly_neighbors = np.argsort(-sims_all)[:top_k_neighbors]

        results[anomaly] = {
            "normal":           int(top_sim_normal),
            "normal_neighbors": anomaly_neighbors.tolist()
        }

    return results

def arg_parse():
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--ds_name', default='COX2', help='dataset name')
    parser.add_argument('--attr', default='node attribute', help='use what node feature, default or node-label')
    parser.add_argument('--h', default=2, help='WL iteration')
    parser.add_argument('--psi1', default=32, help='IK parameter')
    parser.add_argument('--psi2', default=32, help='IK parameter')
    return parser.parse_args()


def main(psi1, psi2, h, adj_list, x_list, class_labels):
    x_list = ik_trans_inne_gpu(x_list, psi1)
    all_embeddings = []
    all_node_embeddings = []

    for i in range(len(adj_list)):
        adj = adj_list[i].toarray()
        x = x_list[i]
        embedding_node = createWlEmbedding_fast(adj, x, h)
        embedding_g = np.mean(embedding_node, axis=0, keepdims=True)
        all_embeddings.append(embedding_g)
        all_node_embeddings.append(embedding_node)

    embedding_all = np.concatenate(all_embeddings, axis=0)
    # Do anomaly detection
    scores, idk_map, model, idkm_mean = idk_anomalyDetector(embedding_all, psi2, t=200)
    auc = roc_auc_score(class_labels, scores)
    print(f"auc:{auc}")

    # cluster_labels = cluster_and_visualize(embedding_all, idk_map, n_clusters=3, cluster_method='spectral', dim_reduction='tsne')
    # vis_embedding(embedding_all, class_labels, dim_reduction='tsne')
    # plot_sorted_scores(scores)
    # visualize_anomaly_by_percent(scores, cluster_labels, embedding_all, anomaly_ratio=0.1)

    # sim_graphs = find_most_similar_normal_samples(idk_map, scores, anomaly_ratio=0.1)
    sim_graphs = find_most_similar_normal_samples_with_neighbors(idk_map, scores, anomaly_ratio=0.05)
    for i, (anomaly, sample_info) in enumerate(sim_graphs.items()):
        if sample_info is not None:
            anomaly_embedding = all_node_embeddings[anomaly]
            # anomaly_embedding = model.transform(anomaly_embedding)
            normal = sample_info['normal']
            normal_embedding = all_node_embeddings[normal]
            # normal_embedding = model.transform(normal_embedding)

            sim_matrix_normal = np.dot(anomaly_embedding, normal_embedding.T)
            # plot_similarity_heatmap(sim_matrix, title="Anomaly vs Normal Similarity")

            neighbor_sim_matrices = {}
            for neighbor in sample_info['normal_neighbors']:
                neighbor_embedding = all_node_embeddings[neighbor]
                sim_matrix_neighbor = np.dot(anomaly_embedding, neighbor_embedding.T)
                neighbor_sim_matrices[neighbor] = sim_matrix_neighbor

            sim_graphs[anomaly] = {
                "normal": normal,
                "normal_neighbors": sample_info['normal_neighbors'],
                "sim_matrix": sim_matrix_normal,
                "neighbor_sim_matrices": neighbor_sim_matrices
            }
    # plot_tsne_with_anomaly_matches(embedding_all, sim_graphs)
    return scores, sim_graphs


if __name__ == '__main__':
    fix_seed = 4
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    args = arg_parse()

    #adj_list, x_list, class_labels = load_data_select(args.ds_name, args.attr, selected_class=1)
    adj_list, x_list, class_labels = load_data(args.ds_name, args.attr)
    label_distribution = Counter(class_labels)
    print(label_distribution)
    scores, sim_graphs = main(args.psi1, args.psi2, args.h, adj_list, x_list, class_labels)
    idx = np.argsort(scores)
    #visualize_matching_motifs(sim_graphs, adj_list, idx, max_pairs=6)
    print(f'j_hat:{find_j(scores)}')
    #visualize_graphs_in_range(adj_list, idx, class_labels, scores, 3, seed=2023)
    #visualize_matching_motifs(sim_graphs, adj_list, idx, max_pairs=5, seed=2021)
    #visualize_neighbors_only(sim_graphs, adj_list, max_rows=2, seed=2020, cmap='Blues')
    #visualize_anomaly_with_neighbors(sim_graphs, adj_list, class_labels, max_rows=5, seed=2021)
    visualize_anomaly_with_neighbors2(sim_graphs, adj_list, class_labels, scores,max_rows=5, seed=2023)

