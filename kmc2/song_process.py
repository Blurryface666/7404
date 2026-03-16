import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from k_mc2 import run_experiment  # Ensure k_mc2 is available; if not, install or define it

def load_song_data(file_path):
    """
    加载 SONG 数据集（逗号分隔，第一列为 ID，后面为特征）
    返回 X: (n_samples, n_features) numpy 数组
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            # 忽略第一列（ID），剩余转为浮点数
            features = [float(x) for x in parts[1:]]
            data.append(features)
    X = np.array(data, dtype=np.float64)
    print(f"Loaded SONG data: {X.shape}")
    return X

# Parameter settings
X = load_song_data("YearPredictionMSD.txt")   # 或者用划分后的训练集

k = 2000
m_values = [1, 10, 50, 100, 200]   # 可根据时间调整
seeds = list(range(50))

results = []
for m in m_values:
    for seed in seeds:
        print(f"Running K-MC2: m={m}, seed={seed}")
        res = run_experiment(X, k, m, seed, dataset_name="SONG")
        results.append(res)

# 保存结果
df = pd.DataFrame(results)
df.to_csv("kmc2_song_results.csv", index=False)

stats = df.groupby('m')['final_cost'].agg(['mean', 'std', 'count'])
print("\n按 m 分组的 final_cost 统计：")
print(stats)

# 按链长 m 分组，计算 dist_computations 的均值
dist_stats = df.groupby('m')['dist_computations'].mean()
print("\n按 m 分组的平均距离计算次数：")
print(dist_stats)

# 可选：输出数值格式更易读（如科学计数法）
print("\n（数值较大，建议使用科学计数法查看）")