import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from kmc2.k_mc2 import run_experiment  # Ensure k_mc2 is available; if not, install or define it

# Load data (integrated from the unused function for simplicity)
file_path = "cod-rna.txt" # Replace with actual path
X_sparse, y = load_svmlight_file(file_path)
X = X_sparse.toarray()  # Convert to dense
print("RNA shape:", X.shape)

# Parameter settings
k = 200
m_values = [1, 10, 50, 100, 200]
n_seeds = 200
seeds = list(range(n_seeds))

results = []
for m in m_values:
    for seed in seeds:
        print(f"Running K-MC2: m={m}, seed={seed}")
        res = run_experiment(X, k, m, seed, dataset_name="RNA")
        results.append(res)

# Save results
df = pd.DataFrame(results)
df.to_csv("kmc2_rna_results.csv", index=False)

stats = df.groupby('m')['final_cost'].agg(['mean', 'std', 'count'])
print("\n按 m 分组的 final_cost 统计：")
print(stats)

# 按链长 m 分组，计算 dist_computations 的均值
dist_stats = df.groupby('m')['dist_computations'].mean()
print("\n按 m 分组的平均距离计算次数：")
print(dist_stats)

# 可选：输出数值格式更易读（如科学计数法）
print("\n（数值较大，建议使用科学计数法查看）")