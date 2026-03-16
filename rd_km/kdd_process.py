# 加载数据（以 KDD 为例）
import numpy as np
import pandas as pd
from rd_kmeans import run_baseline_experiment  # 确保 rd_kmeans.py 中的函数可用

file_path = "bio_train.dat"  # 替换为实际路径
data = np.loadtxt(file_path, dtype=np.float64)
X = data[:, 3:]   # 特征列
k = 200             # KDD 的 k 值

seeds = list(range(200))  # 与 K-MC2 相同种子
results = []

for alg in ['random', 'kmeans++']:
    for seed in seeds:
        print(f"Running {alg}, seed={seed}")
        res = run_baseline_experiment(X, k, alg, seed, dataset_name="KDD")
        results.append(res)

# 保存结果
import pandas as pd
df = pd.DataFrame(results)
df.to_csv("baseline_kdd_results.csv", index=False)

# 显示统计摘要
stats = df.groupby('algorithm')['final_cost'].agg(['mean', 'std', 'count'])
print(stats)