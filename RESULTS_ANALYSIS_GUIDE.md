# Results Analysis Guide

This document explains how to read and evaluate the benchmark outputs in the `results/` folder.
It is written for readers who want to understand what each CSV means, which columns matter, and how to decide whether one clustering method is better than another.

## Benchmark Setup

The benchmark compares six clustering methods:

- `FCM`
- `PCM`
- `PFCM`
- `DPC-FCM`
- `eFCM`
- `DA-FCM`

The benchmark uses eight UCI datasets:

- Iris
- Wine
- Breast Cancer
- Seeds
- Glass
- Ecoli
- Yeast
- Image Segmentation

Each dataset-method pair is run with 30 random seeds:

```text
42, 43, 44, ..., 71
```

The output therefore contains:

```text
8 datasets x 6 methods x 30 seeds = 1440 raw runs
```

All reported means and standard deviations are computed from these 30 runs.

## Result Files

### 1. `all_algorithms_uci_benchmark_summary.csv`

This is the main result file for paper tables and result discussion.

Each row represents:

```text
one dataset + one method
```

Since there are 8 datasets and 6 methods, this file has:

```text
8 x 6 = 48 rows
```

Important columns:

| Column | Meaning |
|---|---|
| `dataset` | Dataset name |
| `method` | Clustering method |
| `n_seeds` | Number of seeds used, should be 30 |
| `seed_min`, `seed_max` | First and last seed used |
| `converged_runs` | Number of runs that converged out of 30 |
| `pc_mean`, `pc_std` | Mean and standard deviation of Partition Coefficient |
| `pe_mean`, `pe_std` | Mean and standard deviation of Partition Entropy |
| `sil_mean`, `sil_std` | Mean and standard deviation of Silhouette score |
| `pur_mean`, `pur_std` | Mean and standard deviation of Purity |
| `nmi_mean`, `nmi_std` | Mean and standard deviation of NMI |
| `ari_mean`, `ari_std` | Mean and standard deviation of ARI |
| `iterations_mean`, `iterations_std` | Average convergence iterations |
| `total_time_s_mean`, `total_time_s_std` | Average runtime in seconds |

The file also contains formatted columns such as:

```text
pc_mean_std
pe_mean_std
ari_mean_std
```

These are convenient for paper tables. They show results as:

```text
mean +/- std
```

Use the numeric `*_mean` and `*_std` columns for calculations, sorting, plotting, or statistical analysis.

Example from the current CSV:

| Dataset | Method | `pc_mean` | `pc_std` | `pe_mean` | `pe_std` |
|---|---|---:|---:|---:|---:|
| Wine | DA-FCM | 0.732631 | 0.000000 | 0.474673 | 0.000000 |
| Wine | FCM | 0.476150 | 0.000000 | 0.894419 | 0.000000 |
| Wine | DPC-FCM | 0.476150 | 0.000000 | 0.894419 | 0.000000 |

How to interpret this row:

- For PC, higher is better, so DA-FCM is better than FCM and DPC-FCM on Wine.
- For PE, lower is better, so DA-FCM is again better.
- The standard deviations are almost zero, which means the result is stable across the 30 seeds.
- FCM and DPC-FCM have identical PC and PE on Wine, meaning density-based initialization alone did not change the final fuzzy partition in this case.

### 2. `all_algorithms_uci_all_runs.csv`

This is the raw evidence file.

Each row represents:

```text
one dataset + one method + one seed
```

This file has:

```text
1440 rows
```

Use this file when you want to inspect individual seed behavior, reproduce the summary statistics, draw boxplots, or debug unstable methods.

Normally, this file is not directly reported in the paper. It supports reproducibility.

Example use:

If someone asks whether the summary row for DA-FCM on Wine really came from 30 runs, filter this file using:

```text
dataset = Wine
method = DA-FCM
```

You should see 30 rows, one for each seed from 42 to 71.
The summary file is produced by averaging those 30 rows.

### 3. `all_algorithms_uci_friedman_tests.csv`

This file answers:

```text
Are the methods significantly different overall?
```

The Friedman test is a non-parametric test used to compare multiple algorithms across multiple datasets.

Important columns:

| Column | Meaning |
|---|---|
| `metric` | Metric being tested |
| `higher_is_better` | Whether a larger value means better performance |
| `block_unit` | Unit used as the statistical block |
| `n_blocks` | Number of blocks, here 8 datasets |
| `n_methods` | Number of compared methods, here 6 |
| `statistic` | Friedman test statistic |
| `p_value` | Statistical p-value |
| `significant_alpha_0_05` | Whether `p_value < 0.05` |

In this benchmark, `block_unit` is:

```text
dataset_mean_over_seeds
```

That means each method is first averaged over 30 seeds on each dataset, then the methods are ranked across datasets.
This is the correct setup for comparing algorithms across datasets.

If `significant_alpha_0_05` is `True`, the methods are significantly different for that metric.
Then you should inspect the Nemenyi file to see which method pairs differ.

Example from the current CSV:

| Metric | Friedman p-value | Significant? |
|---|---:|---|
| PC | 0.000002 | Yes |
| PE | 0.000003 | Yes |
| Silhouette | 0.048948 | Yes |

Interpretation:

- PC and PE have very small p-values, so there is strong evidence that the methods differ on fuzzy membership quality.
- Silhouette is also significant, but only barely below 0.05. Interpret silhouette differences more cautiously.
- Friedman does not say which method is best. It only says that not all methods are equivalent.

### 4. `all_algorithms_uci_average_ranks.csv`

This file gives the average rank of each method for each metric.

Important rule:

```text
Lower mean_rank is better.
```

Example:

```text
metric = pc
method = DA-FCM
mean_rank = 1.0000
```

This means DA-FCM ranked first on PC across the datasets.

Use this file when discussing overall method ranking.

Example from the current CSV:

| Metric | Method | Mean Rank |
|---|---|---:|
| PC | DA-FCM | 1.0000 |
| PC | DPC-FCM | 2.3750 |
| PC | FCM | 2.6250 |
| PC | PCM | 4.8750 |
| PC | eFCM | 4.8750 |
| PC | PFCM | 5.2500 |

Interpretation:

DA-FCM has the best possible average rank on PC.
This means it ranked first for PC on every dataset.

Another example:

| Metric | Method | Mean Rank |
|---|---|---:|
| ARI | FCM | 2.3125 |
| ARI | DA-FCM | 2.3750 |
| ARI | DPC-FCM | 2.6875 |

Interpretation:

DA-FCM is very competitive on ARI, but FCM has a slightly better average ARI rank.
So ARI is not the strongest metric for claiming DA-FCM dominance.

### 5. `all_algorithms_uci_nemenyi_tests.csv`

This file answers:

```text
Which exact method pairs are significantly different?
```

The Nemenyi test is a post-hoc test used after Friedman indicates that not all methods behave the same.

Important columns:

| Column | Meaning |
|---|---|
| `metric` | Metric being compared |
| `method_a`, `method_b` | The compared methods |
| `mean_rank_a`, `mean_rank_b` | Average ranks of the two methods |
| `rank_diff` | Absolute difference between ranks |
| `critical_distance_alpha_0_05` | Minimum rank difference needed for significance |
| `p_value` | Pairwise post-hoc p-value |
| `significant_alpha_0_05` | Whether the pair is significantly different |

Use this file carefully.
A method can have a better mean score but still not be significantly better after Nemenyi correction.

Example from the current CSV:

| Metric | Method A | Method B | Rank Diff | Critical Distance | p-value | Significant? |
|---|---|---|---:|---:|---:|---|
| PC | PFCM | DA-FCM | 4.2500 | 2.6657 | 0.000081 | Yes |
| PC | FCM | DA-FCM | 1.6250 | 2.6657 | 0.506970 | No |
| PC | DPC-FCM | DA-FCM | 1.3750 | 2.6657 | 0.683577 | No |

Interpretation:

- DA-FCM is significantly better than PFCM on PC.
- DA-FCM has a better PC rank than FCM and DPC-FCM, but the rank difference is not large enough to be significant after Nemenyi correction.
- Therefore, it is correct to say DA-FCM is numerically better than FCM/DPC-FCM on PC, but not statistically significantly better under Nemenyi at `alpha = 0.05`.

## Metrics And How To Read Them

| Metric | Full Name | Direction | What It Measures |
|---|---|---|---|
| `PC` | Partition Coefficient | Higher is better | Crispness/quality of fuzzy memberships |
| `PE` | Partition Entropy | Lower is better | Uncertainty of fuzzy memberships |
| `Sil` | Silhouette score | Higher is better | Geometric cluster separation using hard labels |
| `Pur` | Purity | Higher is better | Majority-label correctness inside predicted clusters |
| `NMI` | Normalized Mutual Information | Higher is better | Information agreement between predicted and true labels |
| `ARI` | Adjusted Rand Index | Higher is better | Pairwise label agreement corrected for chance |

For DA-FCM, the most important metrics are:

```text
PC and PE
```

Reason:

DA-FCM changes the fuzzifier and directly modifies the membership matrix.
Therefore, PC and PE most directly evaluate the part of the algorithm that DA-FCM is designed to improve.

External label metrics such as Purity, NMI, and ARI are also important, but they evaluate hard-label agreement after converting fuzzy memberships to cluster labels.

Silhouette is useful, but it can disagree with fuzzy-membership metrics because it only sees hard cluster assignments and geometric separation.

## How To Decide Which Method Is Better

Use this workflow.

### Step 1: Open the summary file

Start with:

```text
results/all_algorithms_uci_benchmark_summary.csv
```

For each dataset, compare method means:

| Metric | Better Method Has |
|---|---|
| `pc_mean` | Higher value |
| `pe_mean` | Lower value |
| `sil_mean` | Higher value |
| `pur_mean` | Higher value |
| `nmi_mean` | Higher value |
| `ari_mean` | Higher value |

### Step 2: Check standard deviation

Look at the matching `*_std` column.

Example:

```text
pc_mean = 0.7721
pc_std = 0.0045
```

This means the average PC is 0.7721 and the method is fairly stable across seeds.

In general:

```text
lower std = more stable
```

A method with high mean and low std is stronger than a method with high mean but high instability.

Example from Iris:

| Dataset | Method | `pc_mean` | `pc_std` | `pur_mean` | `pur_std` |
|---|---|---:|---:|---:|---:|
| Iris | DA-FCM | 0.772105 | 0.004497 | 0.828444 | 0.043976 |
| Iris | FCM | 0.705263 | 0.000000 | 0.840000 | 0.000000 |

Interpretation:

- DA-FCM has better fuzzy membership quality on Iris because PC is higher.
- FCM has slightly better hard-label purity.
- DA-FCM's purity has visible seed variation, so the hard-label assignment is less stable on Iris than FCM.
- This is why it is useful to report both fuzzy metrics and external label metrics.

### Step 3: Check runtime and iterations

Use:

```text
iterations_mean
total_time_s_mean
```

These do not measure clustering quality.
They measure computational cost.

Lower runtime is better if two methods have similar quality.

Example from Wine:

| Method | `pc_mean` | `pe_mean` | `iterations_mean` | `total_time_s_mean` |
|---|---:|---:|---:|---:|
| DA-FCM | 0.732631 | 0.474673 | 16.633333 | 0.002212 |
| FCM | 0.476150 | 0.894419 | 30.900000 | 0.001917 |
| DPC-FCM | 0.476150 | 0.894419 | 22.000000 | 0.002619 |

Interpretation:

- DA-FCM is clearly better on PC and PE.
- DA-FCM also uses fewer iterations than FCM on Wine.
- Runtime is similar at this scale, so quality is the more important comparison here.

### Step 4: Check Friedman test

Open:

```text
results/all_algorithms_uci_friedman_tests.csv
```

If:

```text
p_value < 0.05
```

then there is significant evidence that the compared methods differ for that metric.

In the current result files, all six metrics are significant at `alpha = 0.05`.

### Step 5: Check average ranks

Open:

```text
results/all_algorithms_uci_average_ranks.csv
```

For each metric:

```text
lower mean_rank = better overall rank
```

This helps summarize which method is best across datasets.

### Step 6: Check Nemenyi pairwise results

Open:

```text
results/all_algorithms_uci_nemenyi_tests.csv
```

If:

```text
significant_alpha_0_05 = True
```

then that method pair is significantly different for the given metric.

Do not claim that DA-FCM is significantly better than another method unless the Nemenyi row confirms it.

Example from ARI:

| Metric | Method A | Method B | Rank Diff | p-value | Significant? |
|---|---|---|---:|---:|---|
| ARI | PCM | DA-FCM | 3.6250 | 0.001489 | Yes |
| ARI | FCM | DA-FCM | 0.0625 | 1.000000 | No |

Interpretation:

- DA-FCM is significantly better than PCM on ARI.
- DA-FCM is not significantly different from FCM on ARI.
- In fact, FCM has a slightly better average ARI rank than DA-FCM, so ARI should be discussed carefully.

Example from Silhouette:

| Metric | Method A | Method B | Rank Diff | p-value | Significant? |
|---|---|---|---:|---:|---|
| Silhouette | PCM | DPC-FCM | 2.6875 | 0.046798 | Yes |
| Silhouette | PCM | DA-FCM | 2.1250 | 0.205657 | No |

Interpretation:

- DPC-FCM is significantly better than PCM on silhouette.
- DA-FCM is not significantly better than PCM on silhouette.
- Silhouette is not where DA-FCM shows its strongest advantage.

## Current Result Summary

Based on the generated CSV files:

| Metric | DA-FCM Dataset Wins |
|---|---:|
| PC | 8/8 |
| PE | 7/8 |
| Purity | 6/8 |
| NMI | 5/8 |
| ARI | 5/8 |
| Silhouette | 2/8 |

This means DA-FCM is strongest on fuzzy-membership quality metrics.

The strongest result is:

```text
DA-FCM wins PC on every dataset and PE on seven of eight datasets.
```

This supports the main algorithmic claim because PC and PE directly measure fuzzy partition quality.

### Example Dataset-Level Reading

#### Wine

| Method | PC mean | PE mean | Purity mean | NMI mean | ARI mean |
|---|---:|---:|---:|---:|---:|
| DA-FCM | 0.732631 | 0.474673 | 0.966292 | 0.875894 | 0.897495 |
| FCM | 0.476150 | 0.894419 | 0.966292 | 0.875894 | 0.897495 |
| DPC-FCM | 0.476150 | 0.894419 | 0.966292 | 0.875894 | 0.897495 |
| PCM | 0.334313 | 1.097134 | 0.398876 | 0.000000 | 0.000000 |
| eFCM | 0.333333 | 1.098612 | 0.601124 | 0.470345 | 0.363123 |

Interpretation:

- DA-FCM strongly improves fuzzy membership quality on Wine.
- Hard-label metrics are identical between DA-FCM, FCM, and DPC-FCM.
- This means DA-FCM did not change the final hard clusters on Wine, but it produced much cleaner fuzzy memberships.
- PCM and eFCM perform poorly on Wine, especially on NMI and ARI.

#### Glass

| Method | PC mean | PE mean | Silhouette mean | Purity mean | NMI mean | ARI mean |
|---|---:|---:|---:|---:|---:|---:|
| DA-FCM | 0.412299 | 1.228932 | 0.127595 | 0.598131 | 0.289979 | 0.168692 |
| DPC-FCM | 0.405901 | 1.209204 | 0.220054 | 0.500000 | 0.236206 | 0.131641 |
| FCM | 0.370321 | 1.301133 | 0.215159 | 0.546729 | 0.307817 | 0.155014 |

Interpretation:

- DA-FCM has the best PC and purity among these three methods.
- DPC-FCM has the best PE and silhouette on Glass.
- FCM has the best NMI among these three.
- Therefore, Glass is a mixed dataset: DA-FCM is strong, but not best on every metric.

This is a good example of why the paper should discuss multiple metrics instead of claiming one method dominates everything.

## Average Rank Summary

Lower rank is better.

| Metric | Best Average Rank | Interpretation |
|---|---|---|
| PC | DA-FCM, rank 1.000 | DA-FCM is best overall on membership crispness |
| PE | DA-FCM, rank 1.125 | DA-FCM is best overall on low membership uncertainty |
| Purity | DA-FCM, rank 1.875 | DA-FCM is best overall on purity |
| NMI | DA-FCM, rank 2.000 | DA-FCM is best overall on NMI |
| ARI | FCM rank 2.312, DA-FCM rank 2.375 | FCM is slightly ahead overall, but very close |
| Silhouette | DPC-FCM rank 2.562 | DA-FCM is not the best on silhouette |

## Statistical Test Summary

The Friedman test found significant method-level differences for all six metrics:

| Metric | Friedman p-value | Significant at 0.05 |
|---|---:|---|
| PC | 0.000002 | Yes |
| PE | 0.000003 | Yes |
| Silhouette | 0.048948 | Yes |
| Purity | 0.000020 | Yes |
| NMI | 0.000574 | Yes |
| ARI | 0.000320 | Yes |

The Nemenyi test shows that DA-FCM is significantly better than some weaker baselines, but not necessarily every baseline.

DA-FCM has significant pairwise differences against:

| Metric | DA-FCM Significant Against |
|---|---|
| PC | PCM, PFCM, eFCM |
| PE | PCM, PFCM, eFCM |
| Purity | PCM |
| NMI | PCM |
| ARI | PCM |
| Silhouette | No method significantly worse than DA-FCM |

Important interpretation:

DA-FCM is numerically best on PC and PE, but Nemenyi does not show it is significantly better than FCM or DPC-FCM on those metrics at `alpha = 0.05`.

Therefore, use careful wording.

## Recommended Paper Wording

```text
Across 30 random seeds, DA-FCM achieved the best mean Partition Coefficient on all eight datasets and the best mean Partition Entropy on seven of eight datasets. Friedman tests showed significant differences among methods for all six evaluation metrics. Nemenyi post-hoc analysis confirmed that DA-FCM significantly outperformed weaker baselines such as PCM, PFCM, and eFCM on membership-quality metrics, while differences against FCM and DPC-FCM were not always statistically significant.
```


## Common Mistakes To Avoid

1. Do not use only one seed result.
   Use the 30-seed mean and std from the summary file.

2. Do not say higher PE is better.
   PE is better when lower.

3. Do not report Nemenyi significance from the Friedman file alone.
   Friedman only says at least one method differs.
   Nemenyi tells which pairs differ.

4. Do not claim DA-FCM is best on silhouette.
   It is not. Silhouette is led by other methods overall.

5. Do not ignore standard deviation.
   Stability across seeds matters.

6. Do not compare only runtime.
   Runtime is a cost metric, not a clustering-quality metric.

## Practical Reading Order

For most readers, use this order:

1. `all_algorithms_uci_benchmark_summary.csv`
2. `all_algorithms_uci_average_ranks.csv`
3. `all_algorithms_uci_friedman_tests.csv`
4. `all_algorithms_uci_nemenyi_tests.csv`
5. `all_algorithms_uci_all_runs.csv`

The summary file tells the main story.
The ranks and tests tell whether that story is statistically supported.
The raw runs file proves the results came from all 30 seeds.
