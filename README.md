Github SWIN Transformer: https://github.com/microsoft/Swin-Transformer

Github ConvNeXt: https://github.com/facebookresearch/ConvNeXt

Config hyper-parameters ConvNeXt với C là channels ở các stage, B là stage ratio number:
*   ConvNeXt-T: C = (96, 192, 384, 768), B = (3, 3, 9, 3) 
*   ConvNeXt-S: C = (96, 192, 384, 768), B = (3, 3, 27, 3) 
*   ConvNeXt-B: C = (128, 256, 512, 1024), B = (3, 3, 27, 3)
*   ConvNeXt-L: C = (192, 384, 768, 1536), B = (3, 3, 27, 3) 
*   ConvNeXt-XL: C = (256, 512, 1024, 2048), B = (3, 3, 27, 3)

# Environment
TBD

# How to train?
## 1. Preprocessing TRANCOS dataset
- Using `data_preprocess.py` script
- Open the script and change path to your TRANCOS dataset by updating `dataset_path` variable.
```py
python data_preprocess.py
```
## 2. Train
- Using command: `python train.py`