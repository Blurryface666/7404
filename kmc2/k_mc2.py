import numpy as np
import time
import pandas as pd
from tqdm import tqdm

# ------------------------- 基础距离计算与势函数 -------------------------
def squared_distance_to_centers(x, centers):
    """计算点 x 到中心集 centers 的最小平方距离，并返回距离值及计算次数"""
    # centers: (t, d)
    diff = centers - x
    dist2 = np.sum(diff * diff, axis=1)
    min_dist = np.min(dist2)
    # 本次计算涉及 centers.shape[0] 次点-中心距离
    n_computations = centers.shape[0]
    return min_dist, n_computations

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

# ------------------------- K-MC2 种子算法 -------------------------
def kmc2_seed(X, k, m, seed=0):
    """
    K-MC² 种子选择算法
    参数:
        X: 数据集, shape (n, d)
        k: 聚类数
        m: 马尔可夫链长度
        seed: 随机种子
    返回:
        centers: 所选中心, shape (k, d)
        indices: 中心索引列表
        dist_computations: 总距离计算次数
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    rng = np.random.default_rng(seed)

    # 第一步：随机选择第一个中心
    first_idx = rng.integers(n)
    center_indices = [first_idx]
    centers = [X[first_idx].copy()]
    total_dist_comps = 0  # 累计距离计算次数

    # 后续 k-1 个中心
    for _ in tqdm(range(1, k), desc="K-MC2 seeding"):
        # 当前中心集大小 t = len(centers)
        t = len(centers)
        centers_arr = np.array(centers)

        # 随机初始化链起点 x1
        curr_idx = rng.integers(n)
        curr_dist, comps = squared_distance_to_centers(X[curr_idx], centers_arr)
        total_dist_comps += comps

        # 运行长度为 m 的马尔可夫链
        for _ in range(m):
            cand_idx = rng.integers(n)
            cand_dist, comps = squared_distance_to_centers(X[cand_idx], centers_arr)
            total_dist_comps += comps

            # Metropolis-Hastings 接受概率
            if curr_dist == 0.0:
                accept_prob = 1.0
            else:
                accept_prob = min(cand_dist / curr_dist, 1.0)

            if rng.random() < accept_prob:
                curr_idx = cand_idx
                curr_dist = cand_dist

        # 链最终状态作为新中心
        center_indices.append(curr_idx)
        centers.append(X[curr_idx].copy())

    return np.array(centers), center_indices, total_dist_comps

# ------------------------- 单次实验运行 -------------------------
def run_experiment(X, k, m, seed, dataset_name="KDD"):
    """运行一次 K-MC2 seeding，返回结果字典"""
    start_time = time.time()
    centers, indices, dist_comps = kmc2_seed(X, k, m, seed)
    seeding_time = time.time() - start_time

    # 计算 seeding 后的量化误差
    final_cost = compute_potential(X, centers)

    return {
        "dataset": dataset_name,
        "algorithm": "K-MC2",
        "m": m,
        "seed": seed,
        "final_cost": final_cost,
        "dist_computations": dist_comps,
        "seeding_time": seeding_time   # 可选，仅用于监控
    }

# ------------------------- 主程序：批量运行 -------------------------
if __name__ == "__main__":
    # 加载数据（请根据实际路径修改）
    file_path = r"bio_train.dat"
    data = np.loadtxt(file_path, dtype=np.float64)
    X = data[:, 3:]   # 特征列
    print("数据形状:", X.shape)

    # 参数设置（论文中 KDD 的 k=200）
    k = 200
    m_values = [1, 10, 20, 50, 100, 200]       # 待测链长
    n_seeds = 200                         # 每个设置运行 200 次
    seed_list = list(range(n_seeds))      # 使用 0..199 作为种子

    results = []   # 收集所有结果

    for m in m_values:
        for seed in seed_list:
            print(f"Running K-MC2: m={m}, seed={seed}")
            res = run_experiment(X, k, m, seed, dataset_name="KDD")
            results.append(res)

    # 保存为 CSV
    df = pd.DataFrame(results)
    df.to_csv("kmc2_kdd_results.csv", index=False)
    print("实验结果已保存至 kmc2_kdd_results.csv")

    # 可选：显示统计摘要
    print("\n平均 final_cost 按 m 分组：")
    print(df.groupby("m")["final_cost"].agg(["mean", "std"]))
    print("\n平均 dist_computations 按 m 分组：")
    print(df.groupby("m")["dist_computations"].mean())