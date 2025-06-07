import numpy as np
from sklearn.manifold import TSNE, MDS
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA

def calculate_similarity_matrix(embedding):
    # 计算内积矩阵
    inner_product_matrix = np.dot(embedding, embedding.T)

    # 归一化内积矩阵（可选）
    max_value = np.max(inner_product_matrix)
    normalized_inner_product_matrix = inner_product_matrix / max_value

    return normalized_inner_product_matrix

def calculate_euclidean_distance(embedding):
    # 计算欧氏距离，需要注意欧氏距离与相似度相反，所以需要处理成相似度矩阵
    distance_matrix = euclidean_distances(embedding)
    similarity_matrix = 1 / (1 + distance_matrix)  # 将距离转换为相似度
    return similarity_matrix

def calculate_cosine_similarity(embedding):
    # 计算余弦相似度
    similarity_matrix = cosine_similarity(embedding)
    return similarity_matrix

def visualize_mds(embedding, labels, dataset, h):
    # 对 embedding 进行 TSNE 降维
    mds = MDS(n_components=2, dissimilarity="precomputed")
    matrix = calculate_similarity_matrix(embedding)
    matrix_mds = mds.fit_transform(matrix)

    # 可视化
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        plt.scatter(matrix_mds[labels == label, 0], matrix_mds[labels == label, 1], label=label, s=2, alpha=0.5)
    plt.title('MDS Visualization')
    plt.legend()
    plt.savefig(f'{dataset}_{h}_mds')
    # plt.show()

def visualize_tsne(embedding, labels, ds):
    # 对 embedding 进行 TSNE 降维
    tsne = TSNE(n_components=2)
    matrix_tsne = tsne.fit_transform(embedding)
    # 可视化
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        plt.scatter(matrix_tsne[labels == label, 0], matrix_tsne[labels == label, 1], label=label, s=2, alpha=0.5)
    plt.title(f'{ds} tsne Visualization')
    plt.legend()
    # plt.savefig(f'{dataset}_{h}_tsne')
    plt.show()


def visualize_embeddings(embeddings, labels, ds, method='pca'):
    # 降维处理
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, n_iter=300)
    else:
        raise ValueError("请选择 'pca' 或 'tsne' 作为降维方法")
    reduced = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[labels == 0, 0], reduced[labels == 0, 1],
                c='red', label='Class 0', alpha=0.6, edgecolors='w')
    plt.scatter(reduced[labels == 1, 0], reduced[labels == 1, 1],
                c='blue', label='Class 1', alpha=0.6, edgecolors='w')
    # 添加图例和标题
    plt.title(f'{method.upper()} {ds} Visualization of Embeddings')
    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')
    plt.legend()
    # plt.grid(alpha=0.3)
    plt.show()