import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import numpy as np
import image_utils as iu
from PIL import Image
import random
from torchvision.transforms.functional import to_tensor, to_pil_image
from albumentations import (
    HorizontalFlip,
    Compose,
    Resize,
    Normalize)

NUM_SEQUENCES = 8

class PonyDataset(Dataset):
    def __init__(self, path2data, path2Individual, sequence_number, size, use_memory, transform=None):
        self.sequence_number = sequence_number
        self.size = size
        self.transforms = transform
        count = 0
        self.images = [pp for pp in sorted(os.listdir(path2data)) if 'IMAGE_' in pp]
        self.masks = [pp for pp in sorted(os.listdir(path2data)) if 'MASK_' in pp]
        self.sequences = [] 
        path2Ind = path2Individual
        for pp in self.images:
            individualImg = pp.split('_')[1]
            path2Ind_bg = os.path.join(path2Individual, individualImg)
            listOfImages = sorted(os.listdir(path2Ind_bg))
            path2ImageSequences = []
            for i in range(sequence_number):
                path = os.path.join(path2Ind_bg, listOfImages[count%len(listOfImages)])
                path2ImageSequences.append(path)
                count+=1
            self.sequences.append(path2ImageSequences)
        self.images = [os.path.join(path2data,pp) for pp in self.images]
        self.masks = [os.path.join(path2data,pp) for pp in self.masks]

        self.use_memory = use_memory
        if use_memory:
            self.x_list = []
            self.y_list = []
            self.individual_map = {}
            for idx in range(len(self.images)):
                for i in range(self.sequence_number):
                    imageName = self.sequences[idx][i]
                    if imageName not in self.individual_map:
                        self.individual_map[imageName] = Image.open(imageName).resize((self.size,self.size)).convert('RGB')

            for idx in range(len(self.images)):
                img = Image.open(self.images[idx]).convert('RGB')
                target = Image.open(self.masks[idx])
                
                if self.transforms is not None:
                    augmented= self.transforms(image=np.array(img), mask=np.array(target)[:,:,2])
                    img = augmented['image']
                    target = augmented['mask']
                    
                
                sequences = to_tensor(img)
                target = to_tensor(target).squeeze()
                
                self.x_list.append(sequences)
                self.y_list.append(target)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if self.use_memory:
            sequences = [self.x_list[idx]]
            for i in range(self.sequence_number):
                unique_invd_img = self.individual_map[self.sequences[idx][i]]
                augmented = self.transforms(image=np.array(unique_invd_img))
                imageArray = augmented['image']            
                sequences.append(to_tensor(imageArray))

            sequences = torch.stack(sequences)
            target = self.y_list[idx]
        else:
            img = Image.open(self.images[idx]).convert('RGB')
            target = Image.open(self.masks[idx])
            
            if self.transforms is not None:
                augmented= self.transforms(image=np.array(img), mask=np.array(target)[:,:,2])
                img = augmented['image']
                target = augmented['mask']
                
            sequences = [to_tensor(img)]
            for i in range(self.sequence_number):
                augmented = self.transforms(image=np.array(Image.open(self.sequences[idx][i]).resize((self.size,self.size)).convert('RGB')))
                imageArray = augmented['image']            
                sequences.append(to_tensor(imageArray))

            sequences = torch.stack(sequences)
            target = to_tensor(target).squeeze()
            
        return sequences,target


class PonyGeneratorDataset(Dataset):
    def get_target_label_path(self,path2data, path2Individual, path2postures, number_data):
        target = []
        label = []
        path2Individual = os.path.join(path2data, path2Individual)
        path2postures = os.path.join(path2data, path2postures)

        individualCats = sorted(os.listdir(path2Individual))
        posturesCats = sorted(os.listdir(path2postures))
        
        total = 0
        for posture in posturesCats:
            total = total + len(os.listdir(os.path.join(path2postures, posture)))

        individual_map = {}
        for individual in individualCats:
            individual_map[individual] = sorted(os.listdir(os.path.join(path2Individual, individual)))

        total_partial = 0
        for idx,posture in enumerate(posturesCats):
            posturePath = os.path.join(path2postures, posture)

            posture_image_size = len(sorted(os.listdir(posturePath)))
            if idx != len(posturesCats)-1:
                amount = int(posture_image_size/total*number_data)
                total_partial = total_partial + amount
            else:
                amount = total - total_partial

            for i in range(amount):
                postureImagesPath = sorted(os.listdir(posturePath))
                random_sample = random.choices(population = postureImagesPath, k=2)

                entry = [os.path.join(posturePath, random_sample[0])]
                imageObjective = random_sample[1].split(".")[0]
                assert imageObjective in individual_map, "key "+imageObjective+" not found in map. File Name: "+posturePath+"/"+random_sample[1]
                assert len(individual_map[imageObjective]) > self.sequence_number, "Not enought images for generate sequences in folder: "+imageObjective
                images_sequences = random.choices(population = individual_map[imageObjective], k=self.sequence_number)
                entry = entry + [os.path.join(path2Individual,imageObjective, image) for image in images_sequences]

                target.append(entry)
                label.append(os.path.join(posturePath, random_sample[1]))

        return target, label


    def __init__(self, path2data, path2Individual, path2postures, number_sequences, size, use_memory, number_data, transforms_target_label=None, transforms_individual=None):
        self.sequence_number = number_sequences
        self.size = size
        self.transforms_target_label = transforms_target_label
        self.transforms_individual = transforms_individual
        self.use_memory = use_memory
        self.target_path, self.label_path = self.get_target_label_path(path2data, path2Individual, path2postures, number_data)
        self.path2Image_map = {}

        if use_memory:
            path2Individual = os.path.join(path2data, path2Individual)
            path2postures = os.path.join(path2data, path2postures)

            iu.folder_to_map(path2Individual, self.path2Image_map, lambda image_path: iu.open_image(image_path, self.size), depth=2)
            iu.folder_to_map(path2postures, self.path2Image_map, lambda image_path: iu.open_image(image_path, self.size), depth=2)
        
    def __len__(self):
        return len(self.label_path)
    
    def __getitem__(self, idx):
        target = iu.load_image(self.target_path[idx][0], self.use_memory, self.size, self.path2Image_map)
        label = iu.load_image(self.label_path[idx], self.use_memory, self.size, self.path2Image_map)
    
        if self.transforms_target_label is not None:
            augmented = self.transforms_target_label(image=np.array(target,dtype=np.float32), mask=np.array(label,dtype=np.float32))
            target = augmented['image']
            label = augmented['mask']
        label = iu.interval_mapping(label, 0, 255, 0.0, 1.0)

        sequences = [to_tensor(target)]
        for i in range(self.sequence_number):
            unique_invd_img = iu.load_image(self.target_path[idx][i+1], self.use_memory, self.size, self.path2Image_map)
            if self.transforms_individual is not None:
                augmented = self.transforms_individual(image=np.array(unique_invd_img,dtype=np.float32))
                unique_invd_img = augmented['image']

            sequences.append(to_tensor(unique_invd_img))

        sequences = torch.stack(sequences)
        label = to_tensor(label)
            
        return sequences,label
