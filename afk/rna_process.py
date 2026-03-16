"""
RNA 数据集上仅运行 AFK-MC2 的实验脚本
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from afkmc2 import run_afkmc2_experiment  # 统一接口函数

def load_rna_data(file_path):
    """加载 LIBSVM 格式的 RNA 数据集"""
    print("Loading RNA data...")
    X_sparse, y = load_svmlight_file(file_path)
    X = X_sparse.toarray()  # 转为稠密
    print(f"RNA data loaded: {X.shape}")
    return X

if __name__ == "__main__":
    # 配置
    rna_file = "cod-rna.txt"   # 请修改为实际路径
    k = 200                          # RNA 的聚类数
    seeds = list(range(200))          # 种子 0~14
    m_values = [1, 10, 50, 100, 200]    # 链长

    X = load_rna_data(rna_file)

    results = []
    for m in m_values:
        for seed in seeds:
            print(f"Running AFK-MC2: m={m}, seed={seed}")
            res = run_afkmc2_experiment(X, k, m, seed, "RNA")
            results.append(res)

    df = pd.DataFrame(results)
    output_file = "afkmc2_rna_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # 打印统计摘要
    print("\n=== RNA AFK-MC2 Summary ===")
    stats = df.groupby('m')['final_cost'].agg(['mean', 'std', 'count'])
    print(stats)
    print("\nAverage distance computations:")
    dist_stats = df.groupby('m')['dist_computations'].mean()
    print(dist_stats)