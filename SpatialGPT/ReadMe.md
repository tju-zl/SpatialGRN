# SpatialGPT

## SpatialGPT File Description

| File/Directory     | Description                           |
|-------------------|---------------------------------------|
| `__init__.py`          | Package information |
| `spatialgpt.py`          | Main program file, used to start SpatialGPT |
| `config.py`      | Configuration file, containing basic settings for SpatialGPT |
| `utils.py`   | Collection of helper functions        |
| `data.py`            | Data download, containing all the data preprocess functions used by SpatialGPT |
| `gene2index.csv`   | Gene token embedding index globally       |
| `visaul.py`  | Visaulization functions for SpatialGPT |
| `model.py`  | Model for SpatialGPT |
| `module.py`  | Modules used in model |
| `tokenizer.py`  | Spatial tokens generation program for SpatialGPT |
| `benchmark.py`  | benchmark functions for SpatialGPT |

## Spatial Resolved Transcriptomics Datasets

- The datasets are online avaliable sources.
- The list of dataset is summarized from data provided by multiple work.

| Dataset | Technology | Spots (~) | Genes | Download |
|---|---|---|---|---|
| `12-slice_DLPFC` | 10X Visium | 3460~4789 | 33538 | [Source]([www](https://www.nature.com/articles/s41593-020-00787-0))|
| `HumanBreastCancer` | 10X Visium | 3798 | 36601 | [Source](https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Breast_Cancer_Block_A_Section_1) |
| `12-slice_MouseOlfactoryBulb`| ST | 231~282 | 15284-16675 | [Source](Source)|

