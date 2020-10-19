import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
import matplotlib.pyplot as plt
from albumentations import (
    HorizontalFlip,
    Compose,
    Resize,
    Normalize)

from sklearn.model_selection import ShuffleSplit
from torch.utils.data import Subset

def split_dataset(dataset_train, dataset_val):
    sss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    indices=range(len(dataset_train))

    for train_index, val_index in sss.split(indices):
        print(len(train_index))
        print("-"*10)
        print(len(val_index))
        
    train_ds=Subset(dataset_train,train_index)
    print(len(train_ds))

    val_ds=Subset(dataset_val,val_index)
    print(len(val_ds))

    return train_ds, val_ds

