import numpy as np
import time
from tqdm import tqdm
from sklearn.datasets import make_blobs
import json

class AFKMC2:
    
    def __init__(self, k: int, chain_length: int, random_state: int = 42, verbose: bool = True):
        """
        Parameters:
        k:int,Number of clusters
        chain_length:int,Length of Markov chain (m in paper)
        random_state:int,Random seed
        verbose:bool,Print progress
        """
        self.k = k
        self.m = chain_length
        self.random_state = random_state
        self.verbose = verbose
        self.rng = np.random.default_rng(random_state)
        
        #Key metrics
        self.dist_computations = 0  #Number of distance computations
        self.final_cost = None      #Final quantization error
        self.seeding_time = 0        #Total seeding time
        self.preprocessing_time = 0  #Preprocessing time
        self.loop_time = 0           #Main loop time
        
        self.proposal_dist = None
        self.first_center_idx = None
        self.center_indices = []
        self.centers = []
        
    def _min_sq_dist_to_centers(self, x: np.ndarray, centers_list: list) -> float:
        """
        Compute minimum squared distance from point x to all centers
        and count the number of distance computations
        """
        if len(centers_list) == 0:
            return float('inf')
        
        centers = np.asarray(centers_list)
        diff = centers - x
        dist2 = np.sum(diff * diff, axis=1)
        
        #Key:count every distance computation
        self.dist_computations += len(centers)
        
        return float(np.min(dist2))
    
    def _compute_proposal_distribution(self, X: np.ndarray) -> np.ndarray:
        """
        Compute non-uniform proposal distribution q(x | c1)
        Equation (4): q(x) = 0.5 * A + 0.5 * B
        """
        n = X.shape[0]
        
        if self.verbose:
            print("  Computing proposal distribution...")
        
        #Randomly select first center
        self.first_center_idx = self.rng.integers(n)
        first_center = X[self.first_center_idx]
        
        #Compute squared distances from all points to first center
        diff = X - first_center
        distances_to_first = np.sum(diff * diff, axis=1)
        
        #Preprocessing step computes n distance computations
        self.dist_computations += n
        
        #term A:D²-sampling w.r.t. first center
        total_dist = np.sum(distances_to_first)
        if total_dist > 1e-12:
            term_a = distances_to_first / total_dist
        else:
            term_a = np.ones(n) / n
        
        #term B:uniform
        term_b = np.ones(n) / n
        
        #Final proposal:0.5*A+0.5*B
        proposal = 0.5 * term_a + 0.5 * term_b
        proposal = proposal / np.sum(proposal)
        
        if self.verbose:
            print(f"    Proposal stats: max={np.max(proposal):.6f}, min={np.min(proposal):.6f}")
        
        return proposal
    
    def fit(self, X: np.ndarray) -> tuple[np.ndarray, list[int], dict]:
        """
        Run AFK-MC2 algorithm
        Returns:
        centers : np.ndarray, shape (k, d)
            Selected initial centers
        indices : list
            Indices of centers in the original data
        metrics : dict
            Dictionary containing final_cost, dist_computations, and other metrics
        """
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        
        # Reset counters
        self.dist_computations = 0
        total_start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"AFK-MC2: k={self.k}, m={self.m}, n={n}, d={d}")
            print(f"{'='*60}")
        
        #1.PREPROCESSING
        pre_start = time.time()
        if self.verbose:
            print("\n[Step 1] Preprocessing")
            print("-" * 40)
        
        self.proposal_dist = self._compute_proposal_distribution(X)
        self.preprocessing_time = time.time() - pre_start
        
        #2.MAIN LOOP
        loop_start = time.time()
        if self.verbose:
            print("\n[Step 2] Main Loop")
            print("-" * 40)
        
        #Initialize centers
        self.center_indices = [self.first_center_idx]
        self.centers = [X[self.first_center_idx].copy()]
        
        #Sample remaining k-1 centers
        iterator = range(1, self.k)
        if self.verbose:
            iterator = tqdm(iterator, desc="  Selecting centers", leave=False)
        
        for i in iterator:
            #Sample initial state from proposal distribution
            current_idx = self.rng.choice(n, p=self.proposal_dist)
            current_dist = self._min_sq_dist_to_centers(
                X[current_idx], self.centers
            )
            
            #Markov chain of length m
            for step in range(self.m - 1):
                #Sample candidate point from proposal distribution
                candidate_idx = self.rng.choice(n, p=self.proposal_dist)
                candidate_dist = self._min_sq_dist_to_centers(
                    X[candidate_idx], self.centers
                )
                
                #Metropolis-Hastings acceptance probability
                if current_dist < 1e-12:
                    accept_prob = 1.0
                else:
                    ratio = (candidate_dist * self.proposal_dist[current_idx]) / \
                            (current_dist * self.proposal_dist[candidate_idx])
                    accept_prob = min(ratio, 1.0)
                
                if self.rng.random() < accept_prob:
                    current_idx = candidate_idx
                    current_dist = candidate_dist
            
            #Add selected center
            self.center_indices.append(current_idx)
            self.centers.append(X[current_idx].copy())
        
        self.loop_time = time.time() - loop_start
        self.seeding_time = time.time() - total_start_time
        seeding_dist = self.dist_computations
        #3.COMPUTE FINAL COST
        if self.verbose:
            print("\n[Step 3] Computing final cost...")
        
        self.centers = np.array(self.centers, dtype=np.float64)
        self.final_cost = self._compute_quantization_error(X, self.centers,count_dist=False)
        
        #4.METRICS
        metrics = {
            'final_cost': float(self.final_cost),
            'dist_computations': int(seeding_dist),      # 仅 seeding 过程
            'seeding_time': float(self.seeding_time),
            'preprocessing_time': float(self.preprocessing_time),
            'loop_time': float(self.loop_time),
            'k': self.k,
            'chain_length': self.m,
            'n_samples': n,
            'n_features': d
        }
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("AFK-MC2 Results")
            print(f"{'='*60}")
            print(f"final_cost (Φ_C(X)): {self.final_cost:.4e}")
            print(f"dist_computations: {self.dist_computations:,}")
            print(f"seeding_time: {self.seeding_time:.4f} seconds")
            print(f"  ├─ preprocessing: {self.preprocessing_time:.4f}s")
            print(f"  └─ main loop: {self.loop_time:.4f}s")
        
        return self.centers, self.center_indices, metrics
    
    def _compute_quantization_error(self, X: np.ndarray, centers: np.ndarray, 
                                    batch_size: int = 2000, count_dist: bool = True) -> float:
        X = np.asarray(X, dtype=np.float64)
        centers = np.asarray(centers, dtype=np.float64)
        n = X.shape[0]
        k = centers.shape[0]
        phi = 0.0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_batch = X[start:end]
            batch_n = end - start
            diff = X_batch[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist2 = np.sum(diff * diff, axis=2)
            min_dist2 = np.min(dist2, axis=1)
            phi += np.sum(min_dist2)
            if count_dist:
                self.dist_computations += batch_n * k
        return phi


def run_afkmc2_experimentog():
    print("="*70)
    print("AFK-MC2 Experiment on Synthetic Data")
    print("="*70)
    
    #Generate synthetic data
    n_samples = 5000
    n_features = 20
    true_k = 10
    k = 10
    
    print(f"\nGenerating synthetic data: {n_samples} samples, {n_features} features, {true_k} true clusters")
    X, y_true = make_blobs(
        n_samples=n_samples,
        centers=true_k,
        n_features=n_features,
        random_state=42,
        cluster_std=2.0
    )
    X = X.astype(np.float64)
    
    print(f"Dataset shape: {X.shape}")
    
    #AFK-MC2 with different chain lengths
    chain_lengths = [1, 10, 50, 100, 200]
    n_runs = 5  #Run 5 times for each chain length and take average
    
    print("\n" + "-"*50)
    print("AFK-MC2 with different chain lengths")
    print("-"*50)
    
    all_results = []
    
    for m in chain_lengths:
        print(f"\n  Chain length m = {m}")
        print("  " + "-"*30)
        
        m_results = []
        
        for run in range(n_runs):
            seed = run * 100  #Different random seeds
            
            afk = AFKMC2(k=k, chain_length=m, random_state=seed, verbose=False)
            
            t0 = time.time()
            centers, indices, metrics = afk.fit(X)
            t_afk = time.time() - t0
            
            m_results.append(metrics)
            
            if run == 0:  #Only show detailed results for the first run
                print(f"    Run {run+1}:")
                print(f"      final_cost: {metrics['final_cost']:.4e}")
                print(f"      dist_computations: {metrics['dist_computations']:,}")
                print(f"      time: {t_afk:.4f}s")
        
        #Compute averages
        avg_cost = np.mean([r['final_cost'] for r in m_results])
        avg_dist = np.mean([r['dist_computations'] for r in m_results])
        avg_time = np.mean([r['seeding_time'] for r in m_results])
        std_cost = np.std([r['final_cost'] for r in m_results])
        
        print(f"\n    Average over {n_runs} runs:")
        print(f"      avg final_cost: {avg_cost:.4e} (±{std_cost:.4e})")
        print(f"      avg dist_computations: {avg_dist:.0f}")
        print(f"      avg time: {avg_time:.4f}s")
        
        #Save results
        all_results.append({
            'chain_length': m,
            'n_runs': n_runs,
            'avg_final_cost': float(avg_cost),
            'std_final_cost': float(std_cost),
            'avg_dist_computations': int(avg_dist),
            'avg_time': float(avg_time),
            'individual_runs': m_results
        })
    
    #Save results
    results = {
        'dataset': {
            'name': 'synthetic',
            'n_samples': n_samples,
            'n_features': n_features,
            'true_k': true_k,
            'k': k,
            'cluster_std': 2.0
        },
        'afkmc2_results': all_results
    }
    
    with open('afkmc2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to 'afkmc2_results.json'")
    
    #Concise summary
    print("\n" + "="*70)
    print("AFK-MC2 Results Summary")
    print("="*70)
    print(f"\n{'m':<5} {'Avg Final Cost':<20} {'Std Cost':<15} {'Avg Dist Comp':<15} {'Avg Time(s)':<10}")
    print("-"*70)
    
    for r in all_results:
        print(f"{r['chain_length']:<5} "
              f"{r['avg_final_cost']:<20.4e} "
              f"{r['std_final_cost']:<15.4e} "
              f"{r['avg_dist_computations']:<15,} "
              f"{r['avg_time']:<10.4f}")
    
    return results



def run_afkmc2_experiment(X: np.ndarray, k: int, m: int, seed: int, dataset_name: str) -> dict:
    """
    统一接口：运行一次 AFK-MC2 实验，返回结果字典。
    距离计算次数已按论文要求仅统计 seeding 过程（不含 final_cost 计算）。
    """
    afk = AFKMC2(k=k, chain_length=m, random_state=seed, verbose=False)
    centers, indices, metrics = afk.fit(X)
    return {
        "dataset": dataset_name,
        "algorithm": "AFK-MC2",
        "m": m,
        "seed": seed,
        "final_cost": metrics['final_cost'],
        "dist_computations": metrics['dist_computations'],
        "seeding_time": metrics['seeding_time']
    }

if __name__ == "__main__":
    #AFK-MC2
    results = run_afkmc2_experimentog()
    
    print("\n AFK-MC2 experiments completed!")