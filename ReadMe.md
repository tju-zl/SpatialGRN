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

- [x] **7.24** Total framework.
- [x] **7.25** Data preprocess, especially ATAC-seq.
- [x] **7.26** Model Setting.
- [ ] **7.27-29** Attention interpretation, gene regulatory network(cross attention, modification).
- Toy model. -ok
- GraphSAGE+RandomWalk -ok
- Attention, QKV -awaiting
- visualization -thinking
- [ ] **7.30** Visualization.
- [ ] **7.31** Cross attention, GAS.

Workflow
---

1. read dataset
2. quanlity control
3. embedding construction: 3-D tensor (raw, n-hop neighbors, n times random walk)
4. training
5. GRN inference
6. Network analysis

SpatialGPT Model Zoo
---

We provide the pretrained models of SpatialGPT on various tissue and sequence technology. We recommend using the 'Visium_HumanBrain' model for evaluate the performance of SpatialGPT on the well-known 12-slice DLPFC dataset. If fine-tuning customer dataset, please check the condition list for correct setting the configuration args.

|Model_name|Description|Download|
|--------|--------------|-------|
|Visium_HumanBrain|Pretrained on 12-slice DLPFC dataset|[link](www)|
|Visium_HumanBrain|Pretrained on 12-slice DLPFC dataset|[link](www)|
|Visium_HumanBrain|Pretrained on 12-slice DLPFC dataset|[link](www)|
|Visium_HumanBrain|Pretrained on 12-slice DLPFC dataset|[link](www)|

Spatial Dataset
---

We use the online avaliable dataset to train SpatialGPT.

|Dataset|Description|Download|
|--------|--------------|-------|
|Visium_HumanBrain|Visium, 150, 12-slice DLPFC dataset|[link](www)|
|StereoSeq_MouseEmbryo|Visium, 150, 12-slice DLPFC dataset|[link](www)|
|Visium_HumanBrain|Visium, 150, 12-slice DLPFC dataset|[link](www)|
|Visium_HumanBrain|Visium, 150, 12-slice DLPFC dataset|[link](www)|


Condition List
---

Please provide the correct key for config args. If your dataset is not in the list, please append the keys and values on the config file.

**1. Sequence_technology**

|Seq_tech|Embed_index|
|--------|--------------|
|Visium|1|
|Visium_HD|2|
|Stereo_seq|3|
|Xienium|4|

**2. Spatial_Resolution**

|Spa_res|Embed_index|
|--------|--------------|
|0.2|1|
|2|2|
|10|3|
|30|4|
|50|5|
|100|6|
|150|7|
|300|8|

**Tissue**

|Tissue|Embed_index|
|--------|--------------|
|HumanBrain|1|
|MouseBrain|2|
|HumanLung|3|
|HumanLamb|4|
|HumanTestiny|5|
|MouseEmbryo|6|
|HumanPancries|7|
|MouseLung|8|
