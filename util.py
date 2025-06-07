import networkx as nx
import numpy as np
from utils.iNNE_IK import *
from utils.ik_inne_gpu_v1 import *
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置全局字体为 Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'

def ik_trans_inne(x_train, x_test, psi):
    x_train_all = np.concatenate(x_train, axis=0)
    x_test_all = np.concatenate(x_test, axis=0)
    model = IK_inne_gpu(100, psi)
    model.fit(x_train_all)
    ik_train_ = model.transform(x_train_all)
    ik_test_ = model.transform(x_test_all)

    shapes = [arr.shape for arr in x_train]
    ik_train = []
    start_idx = 0
    for shape in shapes:
        end_idx = start_idx + shape[0]
        reshaped_arr = ik_train_[start_idx:end_idx]
        ik_train.append(reshaped_arr)
        start_idx = end_idx

    shapes = [arr.shape for arr in x_test]
    ik_test = []
    start_idx = 0
    for shape in shapes:
        end_idx = start_idx + shape[0]
        reshaped_arr = ik_test_[start_idx:end_idx]
        ik_test.append(reshaped_arr)
        start_idx = end_idx
    return ik_train, ik_test

def ik_trans_inne_gpu(feat_list, psi):
    feat_all = np.concatenate(feat_list, axis=0)
    model = IK_inne_gpu(200, psi)
    model.fit(feat_all)
    feat_ik = model.transform(feat_all)

    shapes = [arr.shape for arr in feat_list]
    ik = []
    start_idx = 0
    for shape in shapes:
        end_idx = start_idx + shape[0]
        reshaped_arr = feat_ik[start_idx:end_idx]
        ik.append(reshaped_arr)
        start_idx = end_idx

    return ik

def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    return (x - min_val) / (max_val - min_val + 1e-8)

'''画出score'''
def plot_sorted_scores(scores):
    # 输入分数
    # 对分数进行排序
    sorted_scores = sorted(scores)

    # 绘制折线图
    plt.plot(sorted_scores, marker='o')  # 使用圆圈标记每个数据点
    plt.title("Sorted Scores Line Chart")  # 图表标题
    plt.xlabel("Index")  # x轴标签
    plt.ylabel("Score")  # y轴标签
    plt.grid(True)  # 显示网格
    plt.show()

def find_most_similar_normal_samples(features, scores, anomaly_ratio=0.1):
    """
    找到最异常的三个样本，并为每个异常样本寻找最相似的四个正常样本。

    :param features: (N, D) 特征矩阵，每行是一个样本
    :param scores: (N,) 得分数组，分数越低表示越异常
    :param anomaly_ratio: 异常样本占比（0~1），根据得分划分异常与正常样本
    :return: 一个字典，包含最异常的3个样本及其最相似的正常样本索引
    """

    num_samples = len(scores)
    num_anomalies = int(num_samples * anomaly_ratio)  # 计算异常样本数量

    # 按分数升序排序（分数越低越异常）
    sorted_indices = np.argsort(scores)
    anomaly_indices = sorted_indices[:num_anomalies]  # 选择前 num_anomalies 个异常样本
    normal_indices = sorted_indices[num_anomalies:]  # 其余为正常样本

    # 选出最异常的 3 个样本（得分最低的）
    top_anomalies = anomaly_indices[:5]

    results = {}
    for anomaly in top_anomalies:
        anomaly_feature = features[anomaly]  # 获取异常样本的特征
        # 计算该异常样本与所有正常样本的相似度（内积）
        similarities = np.dot(features[normal_indices], anomaly_feature)
        # 选择相似度最高的正常样本
        top_similar_normal = normal_indices[np.argsort(-similarities)[0]]
        results[anomaly] = top_similar_normal
    return results

def find_most_similar_normal_samples_with_neighbors(features, scores, anomaly_ratio=0.1, top_k_anomalies=5, top_k_neighbors=5):
    """
    对每个异常图，找到最相似的一个 normal 图，再找出这个 normal 图最相似的 top_k_neighbors 个图。

    :param features: (N, D) 特征矩阵
    :param scores: (N,) 异常得分（越低越异常）
    :param anomaly_ratio: 异常占比
    :param top_k_anomalies: 最异常图数量
    :param top_k_neighbors: 每个 normal 图的最近邻图数量
    :return: results 字典，包括 anomaly、normal、normal 的邻居图
    """
    num_samples = len(scores)
    num_anomalies = int(num_samples * anomaly_ratio)

    sorted_indices = np.argsort(scores)
    anomaly_indices = sorted_indices[:num_anomalies]
    normal_indices = sorted_indices[num_anomalies:]

    top_anomalies = anomaly_indices[:top_k_anomalies]

    results = {}
    for anomaly in top_anomalies:
        anomaly_feature = features[anomaly]

        # 找出与 anomaly 最相似的一个 normal
        similarities = np.dot(features[normal_indices], anomaly_feature)
        top_similar_normal = normal_indices[np.argsort(-similarities)[0]]


        normal_feature = features[top_similar_normal]
        sims = np.dot(features, normal_feature).astype(np.float64)
        sims[top_similar_normal] = -np.inf
        nearest_indices = np.argsort(-sims)[:top_k_neighbors]

        results[anomaly] = {
            "normal": top_similar_normal,
            "normal_neighbors": nearest_indices.tolist()
        }

    return results


# def visualize_anomaly_with_neighbors(sim_graphs, adj_list, max_rows=5, seed=42):
#     """
#     可视化 anomaly、其最相似 normal 以及 normal 的相似图邻居。
#
#     :param sim_graphs: find_most_similar_normal_samples_with_neighbors 的返回结果字典
#     :param adj_list: 所有图的邻接矩阵列表
#     :param max_rows: 最多展示多少组（每组一个anomaly + 其normal + normal的top邻居图）
#     :param seed: 控制布局一致性
#     """
#     num_rows = min(max_rows, len(sim_graphs))
#     for row_idx, (anomaly_idx, info) in enumerate(sim_graphs.items()):
#         if row_idx >= num_rows:
#             break
#
#         normal_idx = info["normal"]
#         neighbor_indices = info["normal_neighbors"]
#
#         num_cols = 2 + len(neighbor_indices)
#         plt.figure(figsize=(3 * num_cols, 3))
#
#         # Plot anomaly graph (红色)
#         G_anomaly = nx.from_numpy_array(adj_list[anomaly_idx])
#         pos_anomaly = nx.spring_layout(G_anomaly, seed=seed)
#         plt.subplot(1, num_cols, 1)
#         nx.draw(G_anomaly, pos=pos_anomaly, node_color='red', edgecolors='black', with_labels=False, node_size=100)
#         plt.title(f"Anomaly {anomaly_idx}")
#         plt.axis('off')
#
#         # Plot normal graph (蓝色)
#         G_normal = nx.from_numpy_array(adj_list[normal_idx])
#         pos_normal = nx.spring_layout(G_normal, seed=seed)
#         plt.subplot(1, num_cols, 2)
#         nx.draw(G_normal, pos=pos_normal, node_color='blue', edgecolors='black', with_labels=False, node_size=100)
#         plt.title(f"Normal {normal_idx}")
#         plt.axis('off')
#
#         # Plot neighbors of normal (灰色)
#         for i, neighbor_idx in enumerate(neighbor_indices):
#             G_neighbor = nx.from_numpy_array(adj_list[neighbor_idx])
#             pos_neighbor = nx.spring_layout(G_neighbor, seed=seed)
#             plt.subplot(1, num_cols, 3 + i)
#             nx.draw(G_neighbor, pos=pos_neighbor, node_color='gray', edgecolors='black', with_labels=False,
#                     node_size=100)
#             plt.title(f"Neighbor {neighbor_idx}")
#             plt.axis('off')
#
#         plt.suptitle(f"Anomaly {anomaly_idx} and its similar graphs", fontsize=14)
#         plt.tight_layout()
#         plt.show()

def visualize_anomaly_with_neighbors(
        sim_graphs,
        adj_list,
        labels,                 # ← 新增：图的标签列表 (len == len(adj_list))
        max_rows=5,
        seed=42,
        cmap='Blues'):
    """
    可视化 anomaly、其最相似 normal 以及 normal 的相似图邻居，并在标题中显示每张图的 label。

    :param sim_graphs: dict，键为 anomaly_idx，值包含
                       {"normal": int,
                        "normal_neighbors": List[int],
                        "sim_matrix": ndarray,
                        "neighbor_sim_matrices": {idx: ndarray}}
    :param adj_list:   List[np.ndarray]，所有图的邻接矩阵
    :param labels:     List[int/str]，每个图的类别标签
    :param max_rows:   最多可视化多少组
    :param seed:       spring_layout 随机种子，保证布局一致
    :param cmap:       颜色映射
    """
    num_rows = min(max_rows, len(sim_graphs))

    for row_idx, (anomaly_idx, info) in enumerate(sim_graphs.items()):
        if row_idx >= num_rows:
            break

        normal_idx          = info["normal"]
        neighbor_indices    = info["normal_neighbors"]
        sim_matrix          = info["sim_matrix"]                 # (anomaly_nodes, normal_nodes)
        neighbor_sim_mats   = info.get("neighbor_sim_matrices", {})

        num_cols = 2 + len(neighbor_indices)
        plt.figure(figsize=(3 * num_cols, 3))

        # ---------- Anomaly ----------
        G_a   = nx.from_numpy_array(adj_list[anomaly_idx])
        pos_a = nx.spring_layout(G_a, seed=seed)
        a_colors = normalize(np.max(sim_matrix, axis=1))

        plt.subplot(1, num_cols, 1)
        nx.draw(G_a, pos=pos_a,
                node_color=a_colors, cmap=plt.get_cmap(cmap),
                edgecolors='black', node_size=100, with_labels=False,
                vmin=0.0, vmax=1.0)
        plt.title(f"Anomaly {anomaly_idx}  (label={labels[anomaly_idx]})")
        plt.axis('off')

        # ---------- Normal ----------
        G_n   = nx.from_numpy_array(adj_list[normal_idx])
        pos_n = nx.spring_layout(G_n, seed=seed)
        n_colors = normalize(np.max(sim_matrix, axis=0))

        plt.subplot(1, num_cols, 2)
        nx.draw(G_n, pos=pos_n,
                node_color=n_colors, cmap=plt.get_cmap(cmap),
                edgecolors='black', node_size=100, with_labels=False,
                vmin=0.0, vmax=1.0)
        plt.title(f"Normal {normal_idx}  (label={labels[normal_idx]})")
        plt.axis('off')

        # ---------- Neighbors ----------
        for i, nb_idx in enumerate(neighbor_indices):
            G_nb   = nx.from_numpy_array(adj_list[nb_idx])
            pos_nb = nx.spring_layout(G_nb, seed=seed)

            sim_mat_nb = neighbor_sim_mats.get(nb_idx)
            if sim_mat_nb is not None:
                nb_colors = normalize(np.max(sim_mat_nb, axis=0))
            else:
                nb_colors = 'gray'

            plt.subplot(1, num_cols, 3 + i)
            nx.draw(G_nb, pos=pos_nb,
                    node_color=nb_colors,
                    cmap=plt.get_cmap(cmap) if isinstance(nb_colors, np.ndarray) else None,
                    edgecolors='black', node_size=100, with_labels=False,
                    vmin=0.0, vmax=1.0 if isinstance(nb_colors, np.ndarray) else None)
            plt.title(f"Neighbor {nb_idx}\n(label={labels[nb_idx]})")
            plt.axis('off')

        plt.suptitle(f"Anomaly {anomaly_idx} and Related Graphs", fontsize=14)
        plt.tight_layout()
        plt.show()

def visualize_anomaly_with_neighbors2(
        sim_graphs,
        adj_list,
        labels,
        scores,               # ← 新增
        max_rows=5,
        seed=42,
        node_color='skyblue'):
    """
    可视化 anomaly、其最相似 normal 以及 normal 的邻居图。
    节点统一为蓝色，并在标题中写出 label、rank、score。

    :param sim_graphs: 由 find_most_similar_normal_samples_with_neighbors + 相似度
                       计算阶段得到的字典
    :param adj_list:   全部图的邻接矩阵列表
    :param labels:     每个图的类别标签
    :param scores:     每个图的异常分数 (len == len(adj_list))
    :param max_rows:   最多展示多少组
    :param seed:       spring_layout 随机种子
    :param node_color: 所有节点使用的颜色 (默认 skyblue)
    """
    # 计算按分数升序的排名（rank 0 表示最异常 / 分数最低）
    rank_order = np.argsort(scores)                # 从低到高
    rank_dict  = {idx: rank for rank, idx in enumerate(rank_order)}

    num_rows = min(max_rows, len(sim_graphs))
    for row_idx, (anomaly_idx, info) in enumerate(sim_graphs.items()):
        if row_idx >= num_rows:
            break

        normal_idx         = info["normal"]
        neighbor_indices   = info["normal_neighbors"]

        #num_cols = 2 + len(neighbor_indices)
        num_cols = 4
        plt.figure(figsize=(3 * num_cols, 3))

        # ---------- Anomaly ----------
        G_a   = nx.from_numpy_array(adj_list[anomaly_idx])
        pos_a = nx.spring_layout(G_a, seed=seed)
        plt.subplot(1, num_cols, 1)
        nx.draw(G_a, pos=pos_a,
                node_color=node_color, edgecolors='black',
                node_size=100, with_labels=False)
        # plt.title(
        #     f"lbl={labels[anomaly_idx]} | "
        #     f"rank={rank_dict[anomaly_idx]} | score={scores[anomaly_idx]:.4f}", fontsize=24
        # )
        plt.title(
            # f"Class {labels[anomaly_idx]} | "
            f"Rank #1 (gd: Normal)", fontsize=20, fontname='Times New Roman',
        )
        plt.axis('off')

        #---------- Normal ----------
        # G_n   = nx.from_numpy_array(adj_list[normal_idx])
        # pos_n = nx.spring_layout(G_n, seed=seed)
        # plt.subplot(1, num_cols, 2)
        # nx.draw(G_n, pos=pos_n,
        #         node_color=node_color, edgecolors='black',
        #         node_size=100, with_labels=False)
        # plt.title(
        #     f"Rank #{rank_dict[normal_idx]} (gd: Normal)", fontsize=20, fontname='Times New Roman',
        # )
        # plt.axis('off')

        # ---------- Neighbors ----------
        for i, nb_idx in enumerate(neighbor_indices[:3]):
            G_nb   = nx.from_numpy_array(adj_list[nb_idx])
            pos_nb = nx.spring_layout(G_nb, seed=seed)

            plt.subplot(1, num_cols, 2 + i)
            nx.draw(G_nb, pos=pos_nb,
                    node_color=node_color, edgecolors='black',
                    node_size=100, with_labels=False)
            rank = rank_dict[nb_idx]
            # if i == 1:
            #     rank = rank+162
            plt.title(
                # f"Class {labels[nb_idx]} | "
                f"Rank #{rank} (gd: Normal)", fontsize=20, fontname='Times New Roman',
            )
            plt.axis('off')

        plt.suptitle(f"AIDS",
                     fontsize=24, fontname='Times New Roman', y=0.95)
        plt.tight_layout()
        plt.show()

def visualize_neighbors_only(sim_graphs, adj_list, max_rows=5, seed=42, cmap='Blues'):
    """
    仅可视化每个 normal 图的相似邻居图，并用 sim_matrix 的最大值作为节点颜色。

    :param sim_graphs: 包含 anomaly、normal、neighbor_sim_matrices 的字典
    :param adj_list: 所有图的邻接矩阵列表
    :param max_rows: 最多展示多少个 normal 图的邻居组
    :param seed: 控制图布局一致性
    :param cmap: 颜色映射
    """
    num_rows = min(max_rows, len(sim_graphs))
    for row_idx, (_, info) in enumerate(sim_graphs.items()):
        if row_idx >= num_rows:
            break

        neighbor_indices = info["normal_neighbors"]
        neighbor_sim_matrices = info.get("neighbor_sim_matrices", {})

        num_cols = len(neighbor_indices)
        plt.figure(figsize=(3 * num_cols, 3))

        for i, neighbor_idx in enumerate(neighbor_indices):
            G_neighbor = nx.from_numpy_array(adj_list[neighbor_idx])
            pos_neighbor = nx.spring_layout(G_neighbor, seed=seed)

            sim_mat = neighbor_sim_matrices.get(neighbor_idx, None)
            if sim_mat is not None:
                neighbor_colors = normalize(np.max(sim_mat, axis=0))
            else:
                neighbor_colors = 'gray'

            plt.subplot(1, num_cols, i + 1)
            nx.draw(
                G_neighbor, pos=pos_neighbor,
                node_color=neighbor_colors,
                cmap=plt.get_cmap(cmap) if isinstance(neighbor_colors, np.ndarray) else None,
                edgecolors='black',
                node_size=100,
                with_labels=False,
                vmin=0.0, vmax=1.0 if isinstance(neighbor_colors, np.ndarray) else None
            )
            # plt.title(f"Neighbor {neighbor_idx}")
            plt.axis('off')

        #plt.suptitle(f"Similar Neighbors of Normal {info['normal']}", fontsize=14)
        plt.tight_layout()
        plt.show()


def visualize_matching_motifs(sim_graphs, adj_list, idx, max_pairs=10, seed=2023,cmap='Blues'):
    """
    可视化 anomaly-normal 图对，使用“反向相似度”作为颜色值，突出不匹配区域。
    """
    num_pairs = min(max_pairs, len(sim_graphs))
    plt.figure(figsize=(12, 3 * num_pairs))

    for i, (anomaly_idx, data) in enumerate(sim_graphs.items()):
        if i >= num_pairs:
            break

        normal_idx = data['normal']
        sim_matrix = data['sim_matrix']  # shape: (n, m)

        # 🟦 用反向相似度作为颜色（分数越低颜色越深）
        anomaly_colors = 1.0 - normalize(np.max(sim_matrix, axis=1))
        normal_colors = 1.0 - normalize(np.max(sim_matrix, axis=0))

        anomaly_colors = normalize(np.max(sim_matrix, axis=1))
        normal_colors = normalize(np.max(sim_matrix, axis=0))

        # 图结构
        A_anomaly = adj_list[anomaly_idx]
        G_anomaly = nx.from_numpy_array(A_anomaly)

        A_normal = adj_list[normal_idx]
        G_normal = nx.from_numpy_array(A_normal)

        # Plot anomaly
        plt.subplot(num_pairs, 2, i * 2 + 1)
        pos = nx.spring_layout(G_anomaly, seed=seed)
        nx.draw(
            G_anomaly,
            pos=pos,
            node_color=anomaly_colors,
            cmap=plt.get_cmap(cmap),
            node_size=150,
            with_labels=False,
            edgecolors='black',
            vmin=0.0, vmax=1.0
        )
        #plt.title(f"Anomaly {anomaly_idx}")
        plt.axis('off')

        # Plot normal
        plt.subplot(num_pairs, 2, i * 2 + 2)
        pos = nx.spring_layout(G_normal, seed=seed)
        nx.draw(
            G_normal,
            pos=pos,
            node_color=normal_colors,
            cmap=plt.get_cmap(cmap),
            node_size=150,
            with_labels=False,
            edgecolors='black',
            vmin=0.0, vmax=1.0
        )
        #plt.title(f"Normal {normal_idx}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def visualize_matching_motifs2(sim_graphs, adj_list, idx, max_pairs=3, seed=2023, cmap='Blues'):
    """
    每张图单独成图，顺序为 Anomaly_1, Anomaly_2, ..., Normal_1, Normal_2, ...
    """
    num_pairs = min(max_pairs, len(sim_graphs))

    for i, (anomaly_idx, data) in enumerate(list(sim_graphs.items())[:num_pairs]):
        normal_idx = data['normal']
        sim_matrix = data['sim_matrix']  # shape: (n, m)

        # 节点颜色 = 1 - 相似度（匹配差异更亮）
        anomaly_colors = 1.0 - normalize(np.max(sim_matrix, axis=1))
        normal_colors = normalize(np.max(sim_matrix, axis=0))

        # === Anomaly 图 ===
        A_anomaly = adj_list[anomaly_idx]
        G_anomaly = nx.from_numpy_array(A_anomaly)

        plt.figure(figsize=(5, 4))
        pos_a = nx.spring_layout(G_anomaly, seed=seed)
        nx.draw(
            G_anomaly,
            pos=pos_a,
            node_color=anomaly_colors,
            cmap=plt.get_cmap(cmap),
            node_size=150,
            with_labels=False,
            edgecolors='black',
            vmin=0.0, vmax=1.0
        )
        #plt.title(f"Anomaly {anomaly_idx}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # === Normal 图 ===
        A_normal = adj_list[normal_idx]
        G_normal = nx.from_numpy_array(A_normal)

        plt.figure(figsize=(5, 4))
        pos_n = nx.spring_layout(G_normal, seed=seed)
        nx.draw(
            G_normal,
            pos=pos_n,
            node_color=normal_colors,
            cmap=plt.get_cmap(cmap),
            node_size=150,
            with_labels=False,
            edgecolors='black',
            vmin=0.0, vmax=1.0
        )
        #plt.title(f"Normal {normal_idx}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def plot_similarity_heatmap(sim_matrix, title="Similarity Matrix Heatmap", cmap='viridis'):
    plt.figure(figsize=(6, 5))
    plt.imshow(sim_matrix, aspect='auto', cmap=cmap)
    plt.colorbar(label='Similarity')
    plt.title(title)
    plt.xlabel('Normal graph nodes')
    plt.ylabel('Anomaly graph nodes')
    plt.tight_layout()
    plt.show()


def save_embedding(args, embedding_g):
    dir = "embedding"
    os.makedirs(dir, exist_ok=True)
    file = os.path.join(dir, f"{args.ds_name}.npy")
    np.save(file, embedding_g)


def find_j(score):
    # 按分数升序排序（假设分数越高越正常）
    sorted_scores = np.sort(score)
    # 计算 diff
    diff = sorted_scores[-1] - sorted_scores[-100]
    print(f'最正常样本分数:{sorted_scores[-1]}, 第100正常样本分数:{sorted_scores[-100]}')
    # 找到第一异常样本 j_1（即分数最低的样本）
    j_1 = sorted_scores[0]
    # 找到最大的 i 使得 |j_i - j_1| < |diff|
    i = 1
    while i < len(sorted_scores) and abs(sorted_scores[i] - j_1) < abs(diff):
        i += 1
    print(f'最异常样本分数:{j_1}, 第j异常样本分数:{sorted_scores[i-1]}')
    return i - 1

from sklearn.manifold import TSNE
def plot_tsne_with_anomaly_matches(embeddings, sim_graphs, max_anomalies=3):
    """
    使用 t-SNE 降维图嵌入，并可视化前三个 anomaly 及其所有对应的 normal 匹配。

    参数:
    - embeddings: np.array of shape (N, D)，图嵌入
    - sim_graphs: dict, {anomaly_idx: {"normal": [normal_idx1, normal_idx2, ...]}, ...}
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    colors = ['red', 'blue', 'gold']  # 每个 anomaly 一种颜色
    plt.figure(figsize=(8, 6))

    # 所有点初始化为灰色圆点
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], color='lightgray', s=30, label='Others')

    # 依次绘制 anomaly 和其所有 normal
    for i, (anomaly_idx, data) in enumerate(list(sim_graphs.items())[:max_anomalies]):
        color = colors[i]
        anomaly_point = embeddings_2d[anomaly_idx]
        plt.scatter(anomaly_point[0], anomaly_point[1], color=color, marker='x', s=100, label=f'Anomaly {i+1}')

        for normal_idx in data["normal"]:
            normal_point = embeddings_2d[normal_idx]
            plt.scatter(normal_point[0], normal_point[1], color=color, marker='o', s=60)

    plt.legend()
    plt.title("t-SNE: Anomalies and Matched Normals")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def compute_j_hat(score, normal_ref_size=100):
    """
    计算 j_hat：分数越小越异常的情况下，找出第一个异常点明显超出正常范围的索引。

    参数:
    - score: np.array, shape (n,), 异常分数，越小越异常
    - normal_ref_size: int, 参考正常分数的数量（默认前100个最高分）

    返回:
    - j_hat: int
    """
    # 得分越小越异常，因此正常是得分高的，异常是得分低的
    sorted_idx = np.argsort(-score)  # 从大到小，最高分→最低分
    normal_scores = score[sorted_idx[:normal_ref_size]]
    normal_range = abs(normal_scores[0] - normal_scores[-1])

    # 异常节点排序（分数从小到大）
    abnormal_scores = np.sort(score)  # 越靠前越异常

    # 遍历异常节点，找最大的满足条件的 j
    for j in range(len(abnormal_scores)):
        diff = abs(abnormal_scores[j] - abnormal_scores[0])
        if diff >= normal_range:
            return j
    return len(score) - 1