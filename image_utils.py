import os
from PIL import Image
import numpy as np

def folder_to_map(folder_path, dictionary, save_function, depth=1):
    depth = depth-1
    subfolders = os.listdir(folder_path)
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        if depth == 0:
            dictionary[subfolder_path] = save_function(subfolder_path)
        else:
            folder_to_map(subfolder_path, dictionary, save_function,depth)

def open_image(image_path, size):
    return Image.open(image_path).resize((size,size)).convert('RGB')

def load_image(image_path, use_memory, size, dictionary=None):
    if use_memory:
        return dictionary[image_path]
    else:
        return open_image(image_path, size)

def interval_mapping(image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=np.float32)
    return to_min + (scaled * to_range)