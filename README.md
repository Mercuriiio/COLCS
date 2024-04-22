# COLCS
## A novel contrastive-learning-based deep neural network for cancer subtypes identification by integrating multi-omics data
## Introduction
We introduced an end-to-end contrastive-learning-based deep neural network called COLCS, to distinguish the subtypes of the patients with the same tumor. By applying to nine cancer datasets, the experiments show COLCS outperformed the existing methods and can get biologically meaningful cancer subtype labels for cancer-related gene identification.  

<div align=center>
<img src="https://github.com/Mercuriiio/COLCS/blob/main/Framework.png" width="618px">
</div>
## Tutorial
### 1. Download Dataset.
You can download the raw and preprocessed data for the experiments at the following website:  
   
https://drive.google.com/drive/folders/1cROx2J3kbA3VSOEjJeQDKSURD5vEj84-?usp=share_link  
   
Alternatively, you can use your own data. We provide preprocessing methods in [preprocess](https://github.com/Mercuriiio/COLCS/tree/main/preprocess) for h5ad, 10X or csv data types. The data type used during model training and testing is h5ad, so all data is converted to h5ad type during preprocessing. The customized data preprocessing commands are:   
```python
python preprocess/generate_h5ad.py --input_h5ad_path=Path_to_input --save_h5ad_dir=Path_to_Save_Folder
```
where the original input and save path of the data needs to be changed according to your requirements. Here is an example:   
```python
python preprocess/generate_h5ad.py --count_csv_path="./data/original/csv/hnsc_rb1.csv" --save_h5ad_dir="./data/preprocessed/csv/" --filter --norm --log --scale --select_hvg
```
### 2. Training and Testing.
The COLCS training process is divided into two phases, a pre-training phase and a fine-tuning phase, details of which can be found in [train.py](https://github.com/Mercuriiio/COLCS/blob/main/train.py). When performing model training, you need to specify the path of preprocessed data, as well as parameters such as the number of training epochs and the learning rate. The following is an example of a training command.   
```python
python train.py --input_h5ad_path="./data/preprocessed/csv/hnsc_rb1_preprocessed.h5ad" --epochs 100 --lr 1 --batch_size 512 --pcl_r 1024 --cos
```
where ```--epochs``` is the number of pre-training epochs and the number of fine-tuning epochs is set using ```--start_epoch```. ```--lr``` is the initial learning rate and ```--pcl_r``` is the number of negative pairs. ```--cos``` indicates the learning rate decay strategy using the cosine schedule.  
 It is worth noting that both the training and testing modules of COCLS are in the [train.py](https://github.com/Mercuriiio/COLCS/blob/main/train.py) file, and the models from the pre-training phase will be saved in the [checkpoints](https://github.com/Mercuriiio/COLCS/tree/main/checkpoints) folder.
### 3. Results.
COCLS automatically saves the results in the [results](https://github.com/Mercuriiio/COLCS/tree/main/result/COLCS) folder during the test. The model will save the self-supervised feature information of the multi-omics data in the pre-training phase, as well as the clustering results for each epoch.
