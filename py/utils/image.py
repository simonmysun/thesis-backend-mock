# This file includes basic functionality for image processing
# including i/o handling, image augmentation and model input pre-processing
# Author: Stefan Kahl, 2018, Chemnitz University of Technology

import sys
sys.path.append("..")

import os
import copy

import numpy as np
import cv2

######################## CONFIG ##########################
import config as cfg

#Fixed random seed
RANDOM = cfg.getRandomState()

def resetRandomState():
    global RANDOM
    RANDOM = cfg.getRandomState()

########################## I/O ###########################
CACHE = {}
def openImage(path, im_dim=1, cache=False):

    # Do we have path in cache?
    if path in CACHE:
        return CACHE[path]
    
    # Open image
    if im_dim == 3:
        img = cv2.imread(path, 1)
    else:
        img = cv2.imread(path, 0)
        
    # Convert to floats between 0 and 1
    img = np.asarray(img / 255., dtype='float32')

    # Store in cache?
    if cache:
        CACHE[path] = img

    return img

def showImage(img, name='IMAGE', text='', timeout=-1, save=None):

    # Put text on image?
    if len(text) > 0:
        img = addText(img, text)

    cv2.imshow(name, img)
    cv2.waitKey(timeout)

    # Save image?
    if not save == None:
        saveImage(img, save)

scnt = 1
def saveImage(img, path, filename=None):

    global scnt
    img = img.copy()

    # Filename?
    if filename == None:
        filename = 'image_' + str(scnt).zfill(4) + '.png'

    # Path?
    if not os.path.exists(path):
        os.makedirs(path)

    # Normalize?
    if img.max() <= 1.0:
        img *= 255.0

    # Save
    cv2.imwrite(os.path.join(path, filename), img)
    scnt += 1

def addText(img, text):

    h, w = img.shape[:2]
    bs = 30    
    cv2.rectangle(img, (0, h), (w, h - bs), (224, 224, 224), -1)        
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 0.5
    fontcolor = (0, 0, 0)
    fontthickness = 1
    boxsize, _ = cv2.getTextSize(str(text), font, fontscale, fontthickness)
    cv2.putText(img, str(text), (5, h - bs + int((bs - boxsize[1]))), font, fontscale, fontcolor, fontthickness)

    return img

#################### PRE-PROCESSING ######################
def normalize(img, zero_center=False):

    # Preamphasis
    #img = img**3
    #img /= img.max()

    # Normalize
    if not img.min() == 0 and not img.max() == 0:
        img -= img.min()
        img /= img.max()
    else:
        img = img.clip(0, 1)

    # Use values between -1 and 1
    if zero_center:
        img -= 0.5
        img *= 2

    return img

def substractMean(img, clip=True):

    # Only suitable for RGB images
    if len(img.shape) == 3:

        # Normalized image?
        if img.max() <= 1.0:

            img[:, :, 0] -= 0.4850 #B
            img[:, :, 1] -= 0.4579 #G
            img[:, :, 2] -= 0.4076 #R

        else:

            img[:, :, 0] -= 123.680 #B
            img[:, :, 1] -= 116.779 #G
            img[:, :, 2] -= 103.939 #R
        
    else:
        img -= np.mean(img)

    # Clip values
    if clip:
        img = img.clip(0, img.max())

    return img

def prepare(img):

    # ConvNet inputs in Theano are 4D-vectors: (batch size, channels, height, width)
    #img[:64, :] = 0.0
    
    # Add axis if grayscale image
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]

    # Transpose axis, channels = axis 0
    img = np.transpose(img, (2, 0, 1))

    # Add new dimension
    img = np.expand_dims(img, 0)

    return img

######################## RESIZING ########################
def resize(img, width, height, mode='squeeze', interpolation='cubic'):

    if img.shape[:2] == (height, width):
        return img

    if mode == 'crop' or mode == 'cropCenter':
        img = cropCenter(img, width, height, interpolation)
    elif mode == 'cropRandom':
        img = cropRandom(img, width, height, interpolation)
    elif mode == 'fill':
        img = fill(img, width, height, interpolation)
    else:
        img = squeeze(img, width, height, interpolation)

    return img

def squeeze(img, width, height, interpolation):

    if interpolation == 'nearest':
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_CUBIC

    # Squeeze resize: Resize image and ignore aspect ratio    
    return cv2.resize(img, (width, height), interpolation=interpolation)

def cropRandom(img, width, height, interpolation):

    # Random crop: Scale shortest side to minsize, crop with random offset

    # Original image shape
    h, w = img.shape[:2]
    aspect_ratio = float(max(h, w)) / float(min(h, w))

    # Scale original image
    minsize = int(max(width, height) * 1.1)
    if w <= h and w < minsize:
        img = squeeze(img, minsize, int(minsize * aspect_ratio))
    elif h < w and h < minsize:
        img = squeeze(img, int(minsize * aspect_ratio), minsize, interpolation)

    #crop with random offset
    h, w = img.shape[:2]
    top = RANDOM.randint(0, h - height)
    left = RANDOM.randint(0, w - width)
    new_img = img[top:top + height, left:left + width]

    return new_img

def cropCenter(img, width, height, interpolation):

    # Center crop: Scale shortest side, crop longer side

    # Original image shape
    h, w = img.shape[:2]
    aspect_ratio = float(max(h, w)) / float(min(h, w))

    # Scale original image
    if w == h:
        img = squeeze(img, max(width, height), max(width, height), interpolation)
    elif width >= height:
        if h >= w:
            img = squeeze(img, width, int(width * aspect_ratio), interpolation)
        else:            
            img = squeeze(img, int(height * aspect_ratio), height, interpolation)
    else:
        if h >= w:
            img = squeeze(img, int(height / aspect_ratio), height, interpolation)
        else:
            img = squeeze(img, int(height * aspect_ratio), height, interpolation)

    #Crop from original image
    top = (img.shape[0] - height) // 2
    left = (img.shape[1] - width) // 2
    new_img = img[top:top + height, left:left + width]
    
    return new_img

def fill(img, width, height, interpolation):

    # Fill mode: Scale longest side, pad shorter side with noise

    # Determine new shape
    try:
        new_shape = (height, width, img.shape[2])
    except:
        new_shape = (height, width)

    # Allocate array with noise
    new_img = RANDOM.normal(0.0, 1.0, new_shape)

    # Original image shape
    h, w = img.shape[:2]
    aspect_ratio = float(max(h, w)) / float(min(h, w))

    # Scale original image
    if w == h:
        img = squeeze(img, min(width, height), min(width, height), interpolation)
    elif width >= height:
        if h >= w:
            img = squeeze(img, int(height / aspect_ratio), height, interpolation)
        else:
            img = squeeze(img, width, int(width / aspect_ratio), interpolation)
    else:
        if h >= w:
            img = squeeze(img, width, int(width * aspect_ratio), interpolation)            
        else:
            img = squeeze(img, width, int(width / aspect_ratio), interpolation)
            
    # Place original image at center of new image
    top = (height - img.shape[0]) // 2
    left = (width - img.shape[1]) // 2
    new_img[top:top + img.shape[0], left:left + img.shape[1]] = img
    
    return new_img

###################### AUGMENTATION ######################
def augment(img, augmentation={}, count=3, probability=0.5):

    # Make working copy
    augmentations = copy.deepcopy(augmentation)

    # Choose number of augmentations according to count
    # Count = 3 means either 0, 1, 2 or 3 different augmentations
    while(count > 0 and len(augmentations) > 0):

        # Roll the dice if we do augment or not
        if RANDOM.choice([True, False], p=[probability, 1 - probability]):

            # Choose one method
            aug = RANDOM.choice(augmentations.keys())
            
            # Call augementation methods
            if aug == 'flip':
                img = flip(img, augmentations[aug])
            elif aug == 'rotate':
                img = rotate(img, augmentations[aug])
            elif aug == 'zoom':
                img = zoom(img, augmentations[aug])
            elif aug == 'crop':
                if isinstance(augmentations[aug], float):
                    img = crop(img, top=augmentations[aug], left=augmentations[aug], right=augmentations[aug], bottom=augmentations[aug])
                else:
                    img = crop(img, top=augmentations[aug][0], left=augmentations[aug][1], bottom=augmentations[aug][2], right=augmentations[aug][3])
            elif aug == 'roll':
                img = roll(img, vertical=augmentations[aug], horizontal=augmentations[aug])
            elif aug == 'roll_v':
                img = roll(img, vertical=augmentations[aug], horizontal=0)
            elif aug == 'roll_h':
                img = roll(img, vertical=0, horizontal=augmentations[aug])
            elif aug == 'mean':
                img = mean(img, augmentations[aug])
            elif aug == 'noise':
                img = noise(img, augmentations[aug])
            elif aug == 'dropout':
                img = dropout(img, augmentations[aug])
            elif aug == 'blank':
                img = blank(img, augmentations[aug])
            elif aug == 'blackout':
                img = blackout(img, augmentations[aug][0], augmentations[aug][1])
            elif aug == 'blur':
                img = blur(img, augmentations[aug])
            elif aug == 'brightness':
                img = brightness(img, augmentations[aug])
            elif aug == 'multiply':
                img = randomMultiply(img, augmentations[aug])
            elif aug == 'hue':
                img = hue(img, augmentations[aug])
            elif aug == 'lightness':
                img = lightness(img, augmentations[aug])
            elif aug == 'add':
                img = add(img, augmentations[aug])
            else:
                pass

            # Remove key so we avoid duplicate augmentations
            del augmentations[aug]
        
        # Count (even if we did not augment)
        count -= 1
    
    return img    
    
def flip(img, flip_axis=1):
    
    return cv2.flip(img, flip_axis)

def rotate(img, angle, zoom=1.0):

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), RANDOM.uniform(-angle, angle), zoom)
    
    return cv2.warpAffine(img, M,(w, h))

def zoom(img, amount=0.33):

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1 + RANDOM.uniform(0, amount))
    
    return cv2.warpAffine(img, M,(w, h))

def crop(img, top=0.1, left=0.1, bottom=0.1, right=0.1):

    h, w = img.shape[:2]

    t_crop = max(1, int(h * RANDOM.uniform(0, top)))
    l_crop = max(1, int(w * RANDOM.uniform(0, left)))
    b_crop = max(1, int(h * RANDOM.uniform(0, bottom)))
    r_crop = max(1, int(w * RANDOM.uniform(0, right)))

    img = img[t_crop:-b_crop, l_crop:-r_crop]    
    img = squeeze(img, w, h)

    return img

def roll(img, vertical=0.1, horizontal=0.1):

    # Vertical Roll
    img = np.roll(img, int(img.shape[0] * RANDOM.uniform(-vertical, vertical)), axis=0)

    # Horizontal Roll
    img = np.roll(img, int(img.shape[1] * RANDOM.uniform(-horizontal, horizontal)), axis=1)

    return img

def mean(img, normalize=True):

    img = substractMean(img, True)

    if normalize and not img.max() == 0:
        img /= img.max()

    return img

def noise(img, amount=0.05):

    img += RANDOM.normal(0.0, RANDOM.uniform(0, amount**0.5), img.shape)
    img = np.clip(img, 0.0, 1.0)

    return img

def dropout(img, amount=0.25):

    d = RANDOM.choice([0, 1], size=(img.shape), p=[amount, 1 - amount])
    
    return img * d

def blank(img, amount=0.25):

    h = int(img.shape[0] * RANDOM.uniform(amount))
    img[-h:, :] = 0.0

    return img

def blackout(img, p_row=0.1, p_col=0.1):

    #b_width = int(img.shape[1] * amount)
    #b_start = RANDOM.randint(0, img.shape[1] - b_width)

    #img[:, b_start:b_start + b_width] = 0

    row_indicies = RANDOM.choice([0, 1], size=(img.shape[0],), p=[p_row, 1 - p_row])
    col_indicies = RANDOM.choice([0, 1], size=(img.shape[1],), p=[p_col, 1 - p_col])

    img *= row_indicies[None, :]
    img *= col_indicies[:, None]

    return img

def occlude(img, size=0.1, top_left=None, return_pos=False):

    occlusion_size = int(min(img.shape[0], img.shape[1]) * size)
    if top_left == None:
        top_left = (RANDOM.randint(0, img.shape[0] - occlusion_size), RANDOM.randint(0, img.shape[1] - occlusion_size))
    img[top_left[0]:top_left[0] + occlusion_size, top_left[1]:top_left[1] + occlusion_size] = 1

    if return_pos:
        return img, [top_left[0], top_left[0] + occlusion_size, top_left[1], top_left[1] + occlusion_size]
    else:
        return img

def blur(img, kernel_size=3):

     return cv2.blur(img, (kernel_size, kernel_size))

def brightness(img, amount=0.25):

    img *= RANDOM.uniform(1 - amount, 1 + amount)
    img = np.clip(img, 0.0, 1.0)

    return img

def randomMultiply(img, amount=0.25):

    img *= RANDOM.uniform(1 - amount, 1 + amount, size=img.shape)
    img = np.clip(img, 0.0, 1.0)

    return img

def hue(img, amount=0.1):

    try:
        # Only works with BGR images
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] *= RANDOM.uniform(1 - amount, 1 + amount)
        hsv[:, :, 0].clip(0, 360)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)        
    except:
        pass

    return img

def lightness(img, amount=0.25):

    try:
        # Only works with BGR images
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] *= RANDOM.uniform(1 - amount, 1 + amount)
        lab[:, :, 0].clip(0, 255)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except:
        pass

    return img

item_cnt = 0
def add(img, items):

    global item_cnt

    # Choose one item from List
    if item_cnt == 0:
        item_cnt = len(items)
    index = RANDOM.randint(item_cnt)

    # Open and resize image
    img2 = openImage(items[index], cfg.IM_DIM, cache=True)
    img2 = resize(img2, img.shape[1], img.shape[0])

    # Generate random weights
    w1 = RANDOM.uniform(0.25, 0.75)
    w2 = 1 - w1

    # Add images and calculate average
    #img = (img * w1 + img2 * w2) / (w1 + w2)

    img = np.maximum.reduce([img, img2])
    #img = np.maximum.reduce([img, img2 * w1]) / max(1.0, w1)
    ##img = cv2.addWeighted(img, w1, img2, w2, 0)

    #Baseline
    """
    MEAN ACCURACY: 0.529008155761118 
    MEAN MISS RATE: 0.9658322470619416 
    MEAN SPECIES PRECISION: 0.44107719021031716 
    MEAN SPECIES RECALL: 0.21822751608141808 
    MEAN SPECIES F1: 0.26307900783935556 
    MEAN TIME PER SOUNDSCAPE: 42.8021120707194

    img = cv2.addWeighted(img, 0.5, img2, 0.5, 0)

    MEAN ACCURACY: 0.36991473821401294 
    MEAN MISS RATE: 0.9204558822156655 
    MEAN SPECIES PRECISION: 0.26660310462551684 
    MEAN SPECIES RECALL: 0.3747524672343185 
    MEAN SPECIES F1: 0.29295471006983276 
    MEAN TIME PER SOUNDSCAPE: 44.24522608121236

    img = np.maximum.reduce([img, img2])

    MEAN ACCURACY: 0.5506204810365056 
    MEAN MISS RATE: 0.8845227188309867 
    MEAN SPECIES PRECISION: 0.3358570425643824 
    MEAN SPECIES RECALL: 0.4241913902122613 
    MEAN SPECIES F1: 0.35524034010250777 
    MEAN TIME PER SOUNDSCAPE: 43.588239653905234

    img = np.maximum.reduce([img, img2 * w1]) / max(1.0, w1)
    
    MEAN ACCURACY: 0.5703860186016035 
    MEAN MISS RATE: 0.8736331144100358 
    MEAN SPECIES PRECISION: 0.2960456859482347 
    MEAN SPECIES RECALL: 0.4216536160955398 
    MEAN SPECIES F1: 0.3236088668960399 
    MEAN TIME PER SOUNDSCAPE: 45.32771058082581 
        
    """

    return img

def colormap(img):

    # Apply colormap
    img = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_JET)

    return img

def heatmap(img, heat):

    # Make color image
    try:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    except:
        pass

    heatmap = cv2.addWeighted(heat, 0.3, img, 0.7, 0)

    return heatmap

if __name__ == '__main__':

    im_path = '../example/Acadian Flycatcher.png'

    img = openImage(im_path, 1)
    #img = resize(img, 256, 128, mode='squeeze')

    showImage(img)
    
    img = augment(img, {'add': ['/home/sk2487/Code/BirdNET/datasets/us_ger_xc/noise_400_2s/0000_0d63b70c-bc66-4c13-a354_001.png']}, 1, 1.0)
    showImage(img)
    
    
    
