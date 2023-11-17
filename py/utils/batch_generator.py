# -*- coding: utf-8 -*-


import sys
sys.path.append("..")

import math
import numpy as np

import config as cfg
from utils import image

RANDOM = cfg.getRandomState()

#################### IMAGE HANDLING #####################
def loadImageAndTarget(sample, augmentation):

    # Load image
    img = image.openImage(sample[0], cfg.IM_DIM)

    # Resize Image
    img = image.resize(img, cfg.IM_SIZE[0], cfg.IM_SIZE[1], mode=cfg.RESIZE_MODE)

    # Do image Augmentation
    if augmentation:
        img = image.augment(img, cfg.IM_AUGMENTATION, cfg.AUGMENTATION_COUNT, cfg.AUGMENTATION_PROBABILITY)

    # Prepare image for net input
    img = image.normalize(img, cfg.ZERO_CENTERED_NORMALIZATION)
    img = image.prepare(img)

    # Get target
    label = sample[1]
    index = cfg.CLASSES.index(label)
    target = np.zeros((len(cfg.CLASSES)), dtype='float32')
    target[index] = 1.0

    return img, target    

#################### BATCH HANDLING #####################
def batchAugmentation(x, y):

    # Augment batch until desired number of target labels per image is reached
    while cfg.MEAN_TARGETS_PER_IMAGE > 1.0 and np.mean(np.sum(y, axis=1)) < cfg.MEAN_TARGETS_PER_IMAGE:

        # Get two images to combine (we try to prevent i == j (which could result in infinite loops) with excluding ranges)
        i = RANDOM.choice(range(1, x.shape[0] - 1))
        j = RANDOM.choice(range(0, i) + range(i + 1, x.shape[0]))

        #image.showImage(np.transpose(x[i], (1, 2, 0)), 'BEFORE i', 10)
        #image.showImage(np.transpose(x[j], (1, 2, 0)), 'BEFORE j', 10)

        # Combine images and keep the maximum amplitudes
        x[i] = np.maximum.reduce([x[i], x[j]])

        # Re-normalize new image
        x[i] = image.normalize(x[i], cfg.ZERO_CENTERED_NORMALIZATION)

        #image.showImage(np.transpose(x[i], (1, 2, 0)), 'AFTER', -1)

        # Combine targets (makes this task a multi-label classification!)
        y[i] = np.logical_or(y[i], y[j])
    
    return x, y

def getDatasetChunk(split):

    #get batch-sized chunks of image paths
    for i in xrange(0, len(split), cfg.BATCH_SIZE):
        yield split[i:i+cfg.BATCH_SIZE]

def getNextImageBatch(split, augmentation=True): 

    #fill batch
    for chunk in getDatasetChunk(split):

        #allocate numpy arrays for image data and targets
        x_b = np.zeros((cfg.BATCH_SIZE, cfg.IM_DIM, cfg.IM_SIZE[1], cfg.IM_SIZE[0]), dtype='float32')
        y_b = np.zeros((cfg.BATCH_SIZE, len(cfg.CLASSES)), dtype='float32')
        
        ib = 0
        for sample in chunk:

            try:
            
                #load image data and class label from path
                x, y = loadImageAndTarget(sample, augmentation)

                #pack into batch array
                x_b[ib] = x
                y_b[ib] = y
                ib += 1

            except:
                continue

        #trim to actual size
        x_b = x_b[:ib]
        y_b = y_b[:ib]

        # Batch augmentation?
        if augmentation and x_b.shape[0] == cfg.BATCH_SIZE:
            x_b, y_b = batchAugmentation(x_b, y_b)

        #instead of return, we use yield
        yield x_b, y_b

#Loading images with CPU background threads during GPU forward passes saves a lot of time
#Credit: J. Schlüter (https://github.com/Lasagne/Lasagne/issues/12)
def threadedGenerator(generator, num_cached=32):
    
    import Queue
    queue = Queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    #define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    #start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    #run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        try:
            queue.task_done()
            item = queue.get()
        except:
            break

def nextBatch(split, augmentation=True, threaded=True):
    if threaded:
        for x, y in threadedGenerator(getNextImageBatch(split, augmentation)):
            yield x, y
    else:
        for x, y in getNextImageBatch(split, augmentation):
            yield x, y
