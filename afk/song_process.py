"""
SONG 数据集上仅运行 AFK-MC2 的实验脚本
注意：k=2000，计算量较大，可根据需要调整种子数或 m 值。
"""

import numpy as np
import pandas as pd
from afkmc2 import run_afkmc2_experiment

def load_song_data(file_path):
    """加载逗号分隔的 SONG 数据集（第一列为 ID，后90维特征）"""
    print("Loading SONG data...")
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            # 跳过第一列 ID，其余转为浮点数
            features = [float(x) for x in parts[1:]]
            data.append(features)
    X = np.array(data, dtype=np.float64)
    print(f"SONG data loaded: {X.shape}")
    return X

if __name__ == "__main__":
    # 配置
    song_file = "YearPredictionMSD.txt"   # 请修改为实际路径
    k = 2000                          # SONG 的聚类数
    seeds = list(range(50))             # 因计算量大，先只跑 5 个种子
    m_values = [1, 10, 50, 100, 200]                # 先跑较小的 m 值（可后续补充）

    X = load_song_data(song_file)

    results = []
    for m in m_values:
        for seed in seeds:
            print(f"Running AFK-MC2: m={m}, seed={seed}")
            res = run_afkmc2_experiment(X, k, m, seed, "SONG")
            results.append(res)

    df = pd.DataFrame(results)
    output_file = "afkmc2_song_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # 打印统计摘要
    print("\n=== SONG AFK-MC2 Summary ===")
    stats = df.groupby('m')['final_cost'].agg(['mean', 'std', 'count'])
    print(stats)
    print("\nAverage distance computations:")
    dist_stats = df.groupby('m')['dist_computations'].mean()
    print(dist_stats)