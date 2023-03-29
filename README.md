# Training  

Each `python` file in `models` folder can be executed to train and save corresponding model.  
All models use imagenet pre trained weights as initialization.  

Models implemented are :

- Segformer b0 - RGB (`scripts/train_segformers_b0_rgb_pytorch.py`) 
- Segformer b5 - RGB  (`scripts/train_segformers_b5_rgb_pytorch.py`)
- Segformer b0 - 5 channels (`scripts/train_segformers_b0_5c_tensorflow.py`)
- Segformer b5 - 5 channels (`scripts/train_segformers_b5_5c_tensorflow.py`)
- Mask2former swin - base - ade - RGB (`scripts/train_mask2formers_swin_base_rgb_pytorch.py`)
- Mask2former swin - large - ade - RGB (`scripts/train_mask2formers_swin_large_rgb_pytorch.py`)


To train, for example segformer b0 RGB model, open terminal and execute :  
`python scripts/train_segformers_b0_rgb_pytorch.py`

Utility classes, functions and global parameters like folder paths are located in `utils/train.py`.  
Model-specific hyperparameters, like batch size or callbacks must be modified in each model files.  

## Training strategy

Only 600 (randomly selected) images are used for validation.  

- Loss : weighted cross-entropy loss with weight set to 0 for class 13
- Learning Rate = 5e-6
- Batch size varying from 2 to 8, depending model size and memory constraints.
- For 5-channels models, imagenet pre-trained weights of first convolution block are duplicated for 4th and 5th channels
- Augmentation used : geometric, saturation, brightness, contrast)


# Inference

## Ensembling using reconstructed zones

By using georeferencing and projection information of each image, we can easily build VRTs (virtual rasters) of the test set. It is composed of 193 large zones. We predict each zone multiple times by shifting 128 pixels in x and y 4 times each. In that way, we obtain `4*4=16` predictions for each pixel. The final prediction of each pixel is voted by majority. Finally, we split predicted zones into original 512*512 patches. -> 65.03 mIOU.  

Fill data path, choose used models, pixel shift and execute `python  scripts/ensemble_mosaic.py`.  


# Weights

Imagenet pre-trained weights for segformers must be downloaded from here : 
- https://huggingface.co/nvidia/mit-b0
- https://huggingface.co/nvidia/mit-b5
- https://huggingface.co/facebook/mask2former-swin-base-ade-semantic
- https://huggingface.co/facebook/mask2former-swin-large-ade-semantic

Our weights are available here :  
- [Mask2former swin - base - ade - RGB](https://drive.google.com/file/d/1WwOp5uVfneA5PL69mrMlZM0Nrb4kfF5X?usp=share_link)
- [Mask2former swin - large - ade - RGB](https://drive.google.com/drive/folders/1Qtu2YF0VXl0UhO0_7dECgDboyavRiL6r?usp=share_link)
- 
- [Segformer b0 - 5 channels](https://drive.google.com/drive/folders/1Pv1-B-88qyz3A5GKwD3mwj-XiwPHmJxd?usp=share_link)
- [Segformer b5 - 5 channels](https://drive.google.com/drive/folders/1faPQjGJRH70gi-D5RQK_ybPrc6vJFKSG?usp=share_link)
- 
- [Segformer b0 - RGB](https://drive.google.com/drive/folders/1v9j82WJ8PoGGwqsvrD_EkQKP93b6Yh3T?usp=share_link)
- [Segformer b5 - RGB](https://drive.google.com/drive/folders/1nmoqqvlCy-cw_V8xCW6HQM1nMJW_HGs-?usp=share_link)


# Used libraries
- numpy
- tensorflow
- pytorch
- gdal
- albumentation
- transformers


# Specs

Training and inference have been done on a machine with the following specs :  
- Env : Google Colab (pro)
- Memory : 32Gb
- GPU : NVIDIA T4 or V100
