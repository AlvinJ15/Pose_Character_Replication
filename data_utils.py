from PIL import Image
import cv2
import numpy as np
import os
import random

BACKGROUND_FOLDER = 'backgrounds'
IMAGE_FOLDER = 'individual_images'
OUTPUT_FOLDER = 'generated_images'

def get_mask_from_image(image):
    ret, mask = cv2.threshold(image[:, :, 3], 0, 255, cv2.THRESH_BINARY)
    return mask

def save_mask_to_disk(mask, file_name):
    mask = Image.fromarray(mask)
    mask.save(file_name)
    
def resize_image_and_mask(image, mask, percent):
    img_w, img_h = image.size
    img_percent = random.randint(30,100)/100
    newsize = (int(img_w*img_percent),int(img_h*img_percent))
    rescaledImage = image.resize(newsize) 
    rescaledMask = mask.resize(newsize)
    
    return rescaledImage, rescaledMask

def random_offset(background, image, percent):
    bg_w, bg_h = background.size
    randombgw = int(bg_w * percent)
    randombgh = int(bg_h * percent)
    img_w, img_h = image.size
    offset = ((bg_w - img_w) // 2+randombgw, (bg_h - img_h) // 2+randombgh)

    return offset
    
def rotate_image_and_mask(image, mask, degree):
    rotatedImage = image.rotate(degree)
    rotatedMask = mask.rotate(degree)
    
    return rotatedImage, rotatedMask
    
def paste_image_to_background(count, image_folder, image, background, number, folder):
    np_mask = get_mask_from_image(np.array(image))
    
    mask = Image.fromarray(np_mask)
    backgroundShape = np.array(background).shape
    
    for i in range(number):
        newBackground = background.copy()
        
        img_percent = random.randint(20,80)/100
        modifiedImage, modifiedMask = resize_image_and_mask(image, mask, img_percent)
        
        bg_percent = random.randint(0,25)/100
        offset = random_offset(newBackground, modifiedImage, bg_percent)

        if i%5 == 0:
            randomDegree = random.randint(0,90)
            modifiedImage, modifiedMask = rotate_image_and_mask(modifiedImage, modifiedMask, randomDegree)
        
        newBackground.paste(modifiedImage, offset, modifiedImage)
        newBackground.save(os.path.join(folder, 'IMAGE_'+image_folder+'_'+str(count[0])+'.png'))
        background_mask = Image.fromarray(np.zeros(backgroundShape).astype(np.uint8))
        background_mask.paste(modifiedMask, offset, modifiedMask)
        background_mask.save(os.path.join(folder, 'MASK_'+image_folder+'_'+str(count[0])+'.png'))
        count[0]+=1

def rescaled_image_to_background(background, image):
    img_w, img_h = image.size
    bg_w, bg_h = background.size
    if bg_w*0.75 < img_w or bg_h*0.75 < img_h:
        img_percent = bg_w/img_w *0.75
        newsize = (int(img_w*img_percent),int(img_h*img_percent))
        image = image.resize(newsize)
    return image

def rescale_image_to_X(image, size):
    img_w, img_h = image.size
    if img_w >= img_h:
        percent = size/img_w
    else:
        percent = size/img_h
        
    newsize = (int(img_w*percent),int(img_h*percent))
    
    return image.resize(newsize)
def generate_images(background_folder, image_folder, number_samples, output_folder, size=512):
    path2output = os.path.join(OUTPUT_FOLDER, output_folder)
    count = [len(os.listdir(path2output))]
    
    bg_folder = os.path.join(BACKGROUND_FOLDER, background_folder)
    img_folder = os.path.join(IMAGE_FOLDER, image_folder)
    listOfBg = os.listdir(bg_folder)
    listOfImg = os.listdir(img_folder)
    
    for bg_name in listOfBg:
        background = Image.open(os.path.join(bg_folder,bg_name), 'r').resize((size,size))
        
        for img_name in listOfImg:
            image = Image.open(os.path.join(img_folder,img_name), 'r').convert('RGBA')
            image = rescale_image_to_X(image, size=size)
            image = rescaled_image_to_background(background, image)
            paste_image_to_background(count, image_folder, image, background, number_samples, path2output)
    

import cv2
import numpy as np
import random
def get_frames(filename, begin, end, number):
    frame_list = random.sample(range(begin, end), number)
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames=[]
    #frame_list= np.linspace(0, v_len-1, n_frames+1, dtype=np.int16)
    for fn in range(v_len):
        success, frame = v_cap.read()
        if success is False:
            continue
        if (fn in frame_list):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            frames.append(frame)
    
    v_cap.release()
    return frames, v_len

def save_frames(frames, output_folder):
    for i,f in enumerate(frames):
        save_mask_to_disk(f,os.path.join(output_folder,"frame"+str(i)+'.png'))


