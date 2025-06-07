import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
import random


def visualize_graphs_in_range(adj_list, idx, class_labels, scores, a, seed=5):
    """
    可视化前 a 个最正常图（得分最低）和后 a 个最异常图（得分最高）。

    参数:
    - adj_list: list of adjacency matrices
    - idx: np.argsort(scores)，得分从低到高排序后的索引
    - class_labels: 每个图的标签
    - scores: 每个图的异常得分（分数越高越异常）
    - a: 可视化前 a 和后 a 个图
    """
    top_indices = idx[:a]  # 最正常的 a 个图
    bottom_indices = idx[-a:]  # 最异常的 a 个图

    plt.figure(figsize=(5 * a, 8))  # 每列5单位宽，两行共8高

    # 绘制 Top-a（第一行）
    for i, graph_idx in enumerate(top_indices):
        A = adj_list[graph_idx]
        label = class_labels[graph_idx]
        score = scores[graph_idx]
        rank = i

        G = nx.from_numpy_array(A)
        pos = nx.spring_layout(G, seed=seed)

        plt.subplot(2, a, i + 1)
        nx.draw(G, pos=pos, with_labels=False, node_size=150, node_color='salmon', edgecolors='black', linewidths=0.8)
        plt.title(f"Top {i+1} anomaly, class {label}", fontsize=18)
        plt.axis('off')

    # 绘制 Bottom-a（第二行）
    for i, graph_idx in enumerate(bottom_indices):
        A = adj_list[graph_idx]
        label = class_labels[graph_idx]
        score = scores[graph_idx]
        rank = len(scores) - a + i

        G = nx.from_numpy_array(A)
        pos = nx.spring_layout(G, seed=seed)

        plt.subplot(2, a, a + i + 1)
        nx.draw(G, pos=pos, with_labels=False, node_size=150, node_color='skyblue', edgecolors='black', linewidths=0.8)
        plt.title(f"Top {i+1} normal, class {label}", fontsize=18)
        plt.axis('off')

    # plt.suptitle("Top and Bottom Scoring Graphs", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def visualize_node_counts_by_label(adj_list, class_labels):
    """
    可视化数据集中每个图的节点数量，并根据 label 着色区分。

    参数:
    - adj_list: list of adjacency matrices
    - class_labels: list or array of graph labels
    """
    node_counts = [A.shape[0] for A in adj_list]
    labels = np.array(class_labels)
    unique_labels = np.unique(labels)
    colors = ['blue', 'red', 'green', 'orange', 'purple']  # 可扩展更多标签

    plt.figure(figsize=(10, 5))

    for i, label in enumerate(unique_labels):
        indices = np.where(labels == label)[0]
        counts = [node_counts[j] for j in indices]
        plt.scatter(indices, counts, label=f'Label {label}', color=colors[i % len(colors)], s=40)

    plt.xlabel("Graph Index")
    plt.ylabel("Number of Nodes")
    # plt.title("Graph Node Counts by Class Label")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()