# SC2 — Selective Zero-Safe Reconstruction of Single-Cell Expression

SC2 is a research project for **selective reconstruction of sparse single-cell RNA-sequencing expression data**.

The central idea is to treat reconstruction as a selective repair problem. For each currently zero gene-expression entry, SC2 estimates:

1. whether the entry should be repaired; and
2. what positive value should be assigned if a repair is made.

Observed nonzero values are preserved unchanged. A validation-selected repair threshold allows the model to abstain when the evidence for changing a zero entry is weak.

The project is designed around reproducible experiments, fixed evaluation panels, leakage-controlled data preparation, and explicit separation between model development, validation, held-out testing, and external evaluation.

---

## 1. Main research questions

The repository supports experiments addressing questions such as:

- How accurately can synthetically hidden positive expression values be reconstructed?
- Can reconstruction preserve gene-specific variation across cells?
- How conservative can repair remain when deciding whether to change zero entries?
- Does performance transfer across tissues, donors, sequencing technologies, and datasets?
- How does SC2 compare with statistical, nearest-neighbour, low-rank, and probabilistic reconstruction methods?
- Which limitations arise from masking design, gene vocabulary, gene ordering, and domain shift?

The current experiments use a fixed **4,096-gene Ensembl vocabulary** and a curated **250,000-cell CELLxGENE Census corpus**, with external evaluation on Baron pancreas and Zheng68K PBMC data.

---

## 2. Repository principles

Git contains the information needed to understand and reproduce an experiment, while large scientific artifacts remain outside the repository.

### Stored in Git

- source code;
- experiment configurations;
- evaluation code;
- SLURM submission scripts;
- small manifests and metadata;
- data-selection records;
- integrity hashes;
- tests;
- documentation.

### Not stored in Git

- raw sequencing datasets;
- processed expression matrices;
- materialized data shards;
- model checkpoints;
- large prediction files;
- training logs;
- large figures or temporary artifacts.

Large files are stored on the HPC filesystem and are referenced through reproducible paths, manifests, and SHA-256 checks where appropriate.

---

## 3. Computing environments

Development can be performed locally, while data preparation, model training, and large-scale evaluation are intended for the HPC environment.

### Local paths

```text
Repository:
C:/Users/hp/sc2

Data:
C:/Users/hp/sc2_local_data

Outputs:
C:/Users/hp/sc2_local_outputs
```

### HPC paths

```text
Repository:
 /home/3159436/sc2/code

Data:
 /home/3159436/sc2/data

Outputs:
 /home/3159436/sc2/outputs
```

The data and output locations can be overridden through environment variables:

```bash
export SC2_DATA_ROOT=/path/to/data
export SC2_OUTPUT_ROOT=/path/to/outputs
```

Code should use these variables instead of hard-coding machine-specific paths whenever possible.

---

## 4. Environment setup

### Local development

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate sc2
```

For CPU-only local development:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

Local execution is intended mainly for:

- code development;
- unit tests;
- configuration checks;
- small-data debugging;
- plotting and analysis.

Large single-cell matrices and full model runs should not be processed locally unless an appropriate environment has been prepared.

---

## 5. HPC workflow

Large experiments are run through SLURM.

A typical workflow is:

```text
configuration
    ↓
SLURM job
    ↓
training / evaluation
    ↓
output directory
    ↓
metrics and manifests
    ↓
integrity / reproducibility checks
```

Before submitting a scientific run:

1. confirm the active Git commit;
2. confirm that the working tree is clean;
3. verify the intended configuration;
4. verify input paths and manifests;
5. submit the SLURM job;
6. record the resulting job ID and output location.

Large preprocessing or training jobs should not be run directly on login nodes.

---

## 6. Data design

The main corpus is derived from the **CZ CELLxGENE Discover Census**.

The current materialized dataset contains:

| Component | Cells |
|---|---:|
| Training | 200,000 |
| Validation | 25,000 |
| Test | 25,000 |
| **Total** | **250,000** |

The corpus contains multiple tissues, cell types, donors, studies, and sequencing assays.

Cells belonging to the same donor within the same study are kept in the same split to reduce leakage between training and evaluation data.

Gene-selection statistics are computed from the training split only. The resulting ordered **4,096-gene Ensembl vocabulary** is then frozen and reused unchanged for training, validation, internal testing, and external harmonization.

---

## 7. Reconstruction benchmark

SC2 is evaluated using controlled positive-entry masking.

For an expression value known to be positive:

```text
original positive value
        ↓
temporarily replaced by zero
        ↓
model receives corrupted cell
        ↓
model predicts repair probability + repair value
        ↓
prediction compared with the known original value
```

Masking rates of **15%, 30%, and 50%** are used to represent increasing reconstruction difficulty.

This benchmark provides exact targets because the hidden values are known. It should not be interpreted as a physical simulation of lower sequencing depth; future count-level experiments should operate directly on raw molecule counts.

---

## 8. Selective repair

For each candidate zero entry, SC2 produces:

```text
repair probability p
conditional positive estimate μ
expected repair = p × μ
```

A repair is applied only when the gate probability exceeds the selected threshold.

Observed nonzero entries are copied directly into the deployed reconstruction and are therefore preserved by construction.

Thresholds are selected using validation data under a conservative originally-zero-fill budget and are then fixed before held-out and external evaluation.

---

## 9. Evaluation

No single metric is sufficient to describe reconstruction quality.

The project therefore reports several complementary quantities.

### Numerical reconstruction

- Mean-squared error (MSE)
- Mean absolute error (MAE)
- Recovery index

\[
R =
1 -
\frac{\mathrm{MSE}}
{\mathrm{Var}(y_{\mathrm{masked}})}
\]

where:

- \(R=1\) indicates perfect reconstruction;
- \(R=0\) corresponds to the best constant prediction of the hidden targets;
- \(R<0\) indicates performance worse than that constant reference.

### Structural behaviour

- gene-wise Spearman correlation;
- cell-wise Spearman correlation;
- prediction-to-target standard-deviation ratio.

### Selective repair

- AUROC;
- AUPRC;
- recall;
- precision;
- originally-zero fill rate.

### Preservation

Observed nonzero entries are audited to verify that deployment leaves them unchanged.

---

## 10. External evaluation

The current external benchmarks are:

### Zheng68K PBMC

A large public peripheral-blood mononuclear-cell dataset generated with 10x Genomics technology.

### Baron pancreas

A public human pancreatic-islet dataset containing cells from four donors.

These datasets provide complementary biological and technical settings and are evaluated without adapting the trained SC2 model to their outcomes.

External gene mappings, selected cells, masking panels, and availability information are recorded before inference.

---

## 11. Comparator methods

SC2 is evaluated against several reconstruction strategies, including:

- positive-train gene mean;
- positive-train gene median;
- truncated low-rank reconstruction;
- \(k\)-nearest-neighbour reconstruction;
- ALRA;
- scVI.

Hyperparameters are selected using internal validation data only. Once selected, settings are fixed for the held-out internal test and external datasets.

The comparison is reported metric by metric because different reconstruction methods can perform well on different aspects of the task.

---

## 12. Reproducibility

Reproducibility is a core design principle of the project.

Important experimental objects may be accompanied by:

- Git commit identifiers;
- configuration files;
- selected-cell manifests;
- gene-vocabulary manifests;
- deterministic masking panels;
- SHA-256 hashes;
- evaluation receipts;
- fixed randomization procedures.

Scientific results should be traceable to the exact code, configuration, data selection, and evaluation inputs that produced them.

A changed scientific protocol should be treated as a new experiment instead of silently replacing previously evaluated settings.

---

## 13. Testing

Run the repository tests before submitting experiments or merging changes:

```bash
pytest
```

For targeted development, individual test files or modules can be run separately.

Code used for final scientific results should pass the relevant tests before execution on the HPC system.

---

## 14. Development guidelines

When adding a new experiment:

1. define the scientific question;
2. create or update the configuration;
3. keep data selection independent of test outcomes;
4. validate paths and manifests;
5. run small checks before large HPC jobs;
6. record the exact code version;
7. select hyperparameters using validation data only;
8. keep held-out and external results out of the selection process;
9. save compact reproducibility metadata with the result.

Avoid introducing manual changes to evaluation panels, gene mappings, thresholds, or preprocessing after inspecting final results.

---

## 15. Current limitations and future work

The current study uses a controlled reconstruction benchmark and a restricted gene vocabulary. Important extensions include:

- raw-count thinning to simulate reduced molecular sampling more realistically;
- broader gene vocabularies;
- sensitivity tests for gene ordering;
- multi-scan or order-robust state-space models;
- calibration across independent studies;
- broader multi-donor external validation;
- additional reconstruction and imputation baselines;
- downstream biological validation;
- controlled model-capacity scaling.

These extensions are intended to test the boundaries identified by the current experiments before making broader claims about biological generalization.

---

## 16. Data availability

The repository does not redistribute the large source datasets.

The project uses public single-cell resources including:

- CZ CELLxGENE Census;
- 10x Genomics Fresh 68k PBMCs;
- Baron pancreas data from GEO GSE84133.

Users should obtain these datasets from their original sources and follow the corresponding licensing and usage conditions.

---

## 17. Repository

Project repository:

```text
https://github.com/flaviocaroli/sc-2-Striped-Conciliation
```

The repository contains the code, configurations, experiment definitions, and reproducibility material associated with SC2.

---

## 18. Author

**Flavio Caroli**  
Bocconi University
