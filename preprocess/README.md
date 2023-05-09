```python
python preprocess/generate_h5ad.py --input_h5ad_path=Path_to_input --save_h5ad_dir=Path_to_Save_Folder
```
### Example
```python
python preprocess/generate_h5ad.py --count_csv_path="./data/original/csv/hnsc_rb1.csv" --save_h5ad_dir="./data/preprocessed/csv/" --filter --norm --log --scale --select_hvg
```
