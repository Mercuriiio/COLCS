# COLCS
## A novel contrastive-learning-based deep neural network for cancer subtypes identification by integrating multi-omics data
## Introduction
We introduced an end-to-end contrastive-learning-based deep neural network called COLCS, to distinguish the subtypes of the patients with the same tumor. By applying to nine cancer datasets, the experiments show COLCS outperformed the existing methods and can get biologically meaningful cancer subtype labels for cancer-related gene identification.
## Running example
### 1. Download Dataset.
You can download the raw and preprocessed data for the experiments at the following website:
https://drive.google.com/drive/folders/1cROx2J3kbA3VSOEjJeQDKSURD5vEj84-?usp=share_link
Alternatively, you can use your own data. We provide preprocessing methods in [preprocess](https://github.com/Mercuriiio/COLCS/tree/main/preprocess) for h5ad, 10X or csv data types.
### 2. Training.
```python
python train.py --input_h5ad_path="./data/preprocessed/csv/hnsc_rb1_preprocessed.h5ad" --epochs 100 --lr 1 --batch_size 512 --pcl_r 1024 --cos
```
## Citation
