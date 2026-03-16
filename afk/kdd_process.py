"""
AFK-MC2 在 KDD 数据集上的实验脚本（修正版）
直接使用 afk_mc2.py 中提供的统一接口函数。
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

# 导入 AFK-MC2 的统一接口函数
from afkmc2 import AFKMC2, run_afkmc2_experiment  # 确保 afk.py 中的函数可用

# ------------------------------------------------------------
# 数据加载函数
# ------------------------------------------------------------
def load_kdd_data(file_path):
    """
    加载 KDD Cup 2004 蛋白质同源性数据集。
    格式：每行前3列为 [block_id, example_id, label]，后面74维为特征。
    返回特征矩阵 X (n_samples, 74)
    """
    data = np.loadtxt(file_path, dtype=np.float64)
    X = data[:, 3:]   # 特征从第4列开始
    print(f"KDD data loaded: {X.shape}")
    return X

# ------------------------------------------------------------
# 主程序
# ------------------------------------------------------------
if __name__ == "__main__":
    # 配置
    kdd_file = "bio_train.dat"   # 请修改为实际路径
    k = 200                                 # KDD 的聚类数
    seeds = list(range(200))                 # 种子 0~199
    m_values = [1, 10, 50, 100, 200]           # 链长

    # 加载数据
    X = load_kdd_data(kdd_file)

    results = []

    # 运行实验
    for m in m_values:
        for seed in seeds:
            print(f"Running AFK-MC2: m={m}, seed={seed}")
            res = run_afkmc2_experiment(X, k, m, seed, "KDD")
            results.append(res)

    # 保存结果
    df = pd.DataFrame(results)
    output_file = "afkmc2_kdd_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # 显示统计摘要
    print("\n=== KDD AFK-MC2 Results Summary ===")
    stats = df.groupby('m')['final_cost'].agg(['mean', 'std', 'count'])
    print(stats)
    print("\nAverage distance computations:")
    dist_stats = df.groupby('m')['dist_computations'].mean()
    print(dist_stats)