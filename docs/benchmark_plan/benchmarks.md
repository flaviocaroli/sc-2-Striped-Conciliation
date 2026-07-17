# SC2 Benchmark Plan

## Main objective

SC2 is a bulk-guided single-cell repair model. The benchmark suite must test three claims:

1. Dense bulk pretraining improves single-cell reconstruction.
2. Paired bulk calibration improves sample-level realism.
3. The striped Mamba/attention backbone improves gene interaction preservation without losing scalability.

## Benchmark groups

### 1. Single-cell reconstruction

Baselines:
- scGPT
- scFoundation
- GeneMamba
- SC-MAMBA2
- Bubble
- scDTL
- SCRABBLE where applicable

Datasets:
- hPancreas
- MS
- Myeloid
- Myeloid_b
- PBMC12k
- COVID-19
- Baron
- Zheng68K
- Segerstolpe
- current Census lung

Tasks:
- 15%, 30%, 50% masked nonzero recovery
- synthetic dropout recovery
- held-out study reconstruction
- held-out tissue reconstruction

Metrics:
- MSE
- RMSE
- MAE
- per-cell Pearson/Spearman
- per-gene Pearson/Spearman
- marker recovery
- false-positive zero filling
- clustering ARI/NMI after imputation
- annotation macro-F1 after imputation

### 2. Paired bulk-sc consistency

Baselines:
- Bubble
- scDTL
- SCRABBLE
- SC2 bridge controls

Datasets:
- BAL paired benchmark
- any additional matched bulk/sc donor datasets

Tasks:
- donor-held-out paired imputation
- pseudobulk of imputed cells vs real bulk
- over-imputation protection

Metrics:
- pseudobulk-to-bulk MSE
- pseudobulk-to-bulk Pearson/Spearman
- marker preservation
- within-donor heterogeneity preservation
- false-positive zero filling

### 3. Deconvolution and digital cytometry

Baselines:
- BayesPrism
- CIBERSORTx
- Bisque
- hspe
- omnideconv methods

Datasets:
- BAL paired benchmark
- DLPFC benchmark

Tasks:
- cell-type fraction prediction
- cell-type-specific expression
- synthetic 1000-cell matrix generation from bulk

Metrics:
- per-type RMSE
- per-type Pearson/Spearman
- Jensen-Shannon divergence
- rare-type RMSE
- calibration error
- discriminator real-vs-generated accuracy

### 4. Adversarial robustness

Stress tests:
- 70% masking
- rare cell-type downsampling
- true-zero protection
- batch/platform shift
- tissue-held-out shift
- donor-held-out shift
- gene-order artifact test
- bulk-reference mismatch test
- mean-collapse test

## Reported numbers policy

Reported numbers from papers can be used only as literature context. The main SC2 table must use controlled reruns or official checkpoints on the same splits, preprocessing, gene universe, and metrics.