import numpy as np
import time
from tqdm import tqdm


def compute_potential(X, centers, batch_size=2000):
    """计算量化误差 Phi_C(X)"""
    X = np.asarray(X, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    n = X.shape[0]
    phi = 0.0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_batch = X[start:end]
        diff = X_batch[:, None, :] - centers[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        min_dist2 = np.min(dist2, axis=1)
        phi += np.sum(min_dist2)
    return phi

def random_seed(X: np.ndarray, k: int, seed: int = 0):
    """
    随机选择 k 个中心。
    返回:
        centers: (k, d) 数组
        indices: 中心索引列表
        dist_computations: seeding 过程中距离计算次数（此处为0）
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    rng = np.random.default_rng(seed)
    center_indices = rng.choice(n, size=k, replace=False).tolist()
    centers = X[center_indices].copy()
    dist_computations = 0  # 随机选择不需要距离计算
    return centers, center_indices, dist_computations

# ------------------------------------------------------------
# k-means++ seeding
# ------------------------------------------------------------
def kmeans_plus_plus_seed(X: np.ndarray, k: int, seed: int = 0):
    """
    k-means++ 种子选择算法。
    返回:
        centers: (k, d) 数组
        indices: 中心索引列表
        dist_computations: seeding 过程中距离计算次数（每次添加中心时计算所有点到新中心的距离）
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    rng = np.random.default_rng(seed)
    dist_computations = 0

    # 第一个中心随机
    first_idx = int(rng.integers(n))
    center_indices = [first_idx]
    centers = [X[first_idx].copy()]

    min_dists = np.full(n, np.inf, dtype=np.float64)

    for i in range(1, k):
        new_center = centers[-1]
        diff = X - new_center
        new_dists = np.sum(diff * diff, axis=1)
        dist_computations += n  # 每个点计算到新中心的距离

        min_dists = np.minimum(min_dists, new_dists)

        total = min_dists.sum()
        if total == 0.0:
            next_idx = int(rng.integers(n))
        else:
            probs = min_dists / total
            next_idx = int(rng.choice(n, p=probs))

        center_indices.append(next_idx)
        centers.append(X[next_idx].copy())

    centers = np.array(centers, dtype=np.float64)
    return centers, center_indices, dist_computations

# ------------------------------------------------------------
# 统一实验运行函数（与 K-MC2 的 run_experiment 类似）
# ------------------------------------------------------------
def run_baseline_experiment(X, k, algorithm, seed, dataset_name):
    """
    运行 baseline 算法（random 或 k-means++）的一次实验。
    返回包含结果字段的字典。
    """
    start_time = time.time()
    if algorithm == 'random':
        centers, indices, dist_comps = random_seed(X, k, seed)
    elif algorithm == 'kmeans++':
        centers, indices, dist_comps = kmeans_plus_plus_seed(X, k, seed)
    else:
        raise ValueError("algorithm must be 'random' or 'kmeans++'")
    seeding_time = time.time() - start_time

    # 计算 seeding 后的量化误差
    final_cost = compute_potential(X, centers)

    return {
        "dataset": dataset_name,
        "algorithm": algorithm,
        "m": 0,  # 占位，无意义
        "seed": seed,
        "final_cost": final_cost,
        "dist_computations": dist_comps,
        "seeding_time": seeding_time
    }