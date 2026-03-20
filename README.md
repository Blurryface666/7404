# COMP7404 Group8 – Code Repository for Clustering Seeding Algorithms

This repository contains our implementation of several clustering seeding algorithms used for experimental comparison:

- **AFK-MC²**
- **K-MC²**
- **Baselines**
  - Random seeding
  - k-means++ seeding

The code is organized to support experiments on multiple datasets, including **KDD**, **RNA**, and **SONG**.

---

## Repository Overview

    7404/
    ├── afk/
    │   ├── afkmc2.py
    │   ├── kdd_process.py
    │   ├── rna_process.py
    │   └── song_process.py
    ├── kmc2/
    │   ├── k_mc2.py
    │   ├── rna_process.py
    │   └── song_process.py
    ├── rd_km/
    │   ├── rd_kmeans.py
    │   ├── kdd_process.py
    │   ├── rna_process.py
    │   └── song_process.py
    └── README.md

## Project Goal

The goal of this repository is to provide a working implementation of different clustering initialization (seeding) methods and compare them in terms of:

- final clustering cost / quantization error

- number of distance computations

- seeding time

The implementation focuses on reproducible experimental evaluation across different datasets and different Markov chain lengths.

## Implemented Algorithms
### 1. AFK-MC²

Location:
```python
afk/afkmc2.py
```
### 2. K-MC²
Location:
```python
kmc2/k_mc2.py
```
### 3. Baselines
Location:
```python
rd_km/rd_kmeans.py
```
Includes:
- random seeding

- k-means++ seeding

## Supported Datasets

The repository supports the following datasets:

1. KDD

3. RNA

5. SONG

**Important Note**

Datasets are NOT included in this repository. You must download them yourself and update file paths in the scripts.

## Environment Requirements

Recommended:

- Python 3.9+

- numpy

- pandas

- scikit-learn

- tqdm

Install dependencies:
```python
pip install numpy pandas scikit-learn tqdm
```
## How to Run
### 1. AFK-MC²
```bash
cd afk
python kdd_process.py
python rna_process.py
python song_process.py
```
### 2. K-MC²
```bash
cd kmc2
python kdd_process.py
python rna_process.py
python song_process.py
```
### 3. Baselines
```bash
cd rd_km
python kdd_process.py
python rna_process.py
python song_process.py
```
## Output

Each experiment generates CSV files such as:

`afkmc2_kdd_results.csv`

`afkmc2_rna_results.csv`

`afkmc2_song_results.csv`

## Authors

Chen Shuqi, Wang Kerui, Wang Yuhui, Zhao Jiayi

## Repository Link

https://github.com/Blurryface666/7404

## Reference

This implementation is based on the following paper:

**Fast and Provably Good Seedings for k-Means**  
Olivier Bachem, Mario Lucic, S. Hamed Hassani, Andreas Krause  

Proceedings of the 30th International Conference on Neural Information Processing Systems (NeurIPS), 2016.
