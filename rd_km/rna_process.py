"""
RNA 数据集实验脚本
运行 random、k-means++ 和 K-MC2 的 seeding，并保存结果。
"""

import numpy as np
import pandas as pd
import time
from tqdm import tqdm

# 假设以下函数已定义在 baseline.py 和 k_mc2.py 中
# 如果这些函数直接写在此脚本中，请将对应代码复制过来
from rd_kmeans import run_baseline_experiment

# ------------------------------------------------------------
# 数据加载函数
# ------------------------------------------------------------
def load_rna_data(file_path):
    """加载 LIBSVM 格式的 RNA 数据集"""
    from sklearn.datasets import load_svmlight_file
    print("Loading RNA data...")
    X_sparse, y = load_svmlight_file(file_path)
    X = X_sparse.toarray()  # 转为稠密（若内存不足可考虑保留稀疏）
    print(f"RNA data loaded: {X.shape}, labels shape: {y.shape}")
    return X

# ------------------------------------------------------------
# 主程序
# ------------------------------------------------------------
if __name__ == "__main__":
    # ========== 配置 ==========
    rna_file = "cod-rna.txt"          # 请修改为实际路径
    k = 200                                 # RNA 的聚类数
    seeds = list(range(200))                # 种子 0~199

    # 加载数据
    X = load_rna_data(rna_file)

    results = []

    # 1. Random seeding
    for seed in seeds:
        print(f"Running Random, seed={seed}")
        res = run_baseline_experiment(X, k, 'random', seed, dataset_name="RNA")
        results.append(res)

    # 2. k-means++ seeding
    for seed in seeds:
        print(f"Running k-means++, seed={seed}")
        res = run_baseline_experiment(X, k, 'kmeans++', seed, dataset_name="RNA")
        results.append(res)


    # 保存结果
    df = pd.DataFrame(results)
    output_file = "baseline_rna_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nRNA results saved to {output_file}")

    # 显示统计摘要
    print("\n=== RNA Summary ===")
    stats = df.groupby(['algorithm', 'm'])['final_cost'].agg(['mean', 'std', 'count'])
    print(stats)
    print("\nAverage distance computations:")
    dist_stats = df.groupby(['algorithm', 'm'])['dist_computations'].mean()
    print(dist_stats)