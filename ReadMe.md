**SpatialGRN**: Spot-Specific Gene Regulatory Network Inference Method using Spatial Multi-Omics Data
===

Using Attention interpretation mechanism to inference GRN.

Description
---

- Date: 2024-07-24
- Developer: Lei Zhang
- Project: SpatialGRN

Framework
---

- Data: spatial ATAC-seq and SRT
- Preprocessing: Gene Activity Score and multiscale robust gene embedding
- Model Construction: Cross attention

Time Line
---

- [ ] **7.24** Total framework.
- [ ] **7.25** Data preprocess, especially ATAC-seq.
- [ ] **7.26** Model Setting.
- [ ] **7.27** Attention interpretation.
- [ ] **7.28** Benchmark of the competing methods.

Workflow
---

1. read dataset
2. quanlity control
3. embedding construction: 3-D tensor (raw, n-hop neighbors, n times random walk)
4. training
5. GRN inference
6. Network analysis
7. 