# Native Mamba Bridge Results

## Current internal benchmark

Dataset:
- ARCHS4 lung large bulk
- Census lung large sc
- Census pseudobulk
- 4096 shared genes

### Native original-like

Parameters: 2,176,259

Internal eval:
- bulk MSE: 0.346578
- sc MSE: 0.025160
- pseudobulk MSE: 0.020638

GTEx lung external:
- MSE: 0.588898
- MAE: 0.310675

### Native GeneMamba-like

Parameters: 2,307,843

Internal eval:
- bulk MSE: 0.392366
- sc MSE: 0.027097
- pseudobulk MSE: 0.021799

### Native SC-Mamba2-like

Parameters: 1,981,187

Internal eval:
- bulk MSE: 0.313887
- sc MSE: 0.022871
- pseudobulk MSE: 0.013546

GTEx lung external:
- MSE: 0.655361
- MAE: 0.318114

## Interpretation

Internal winner:
- SC-Mamba2-like

External GTEx winner:
- original-like

Current decision:
- keep both original-like and SC-Mamba2-like as controls
- pause GeneMamba-like unless rank-order ablation is needed
- next model should be SC2-mini striped Mamba/attention