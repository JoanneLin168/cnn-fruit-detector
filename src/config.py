import torch
import os

BATCH_SIZE = 4 # increase / decrease according to GPU memory
RESIZE_TO = 512 # resize the image for training and transforms
NUM_EPOCHS = 6 # number of epochs to train for
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# training/validation images and XML files directory
TRAIN_DIR = '../data/train'
VALID_DIR = '../data/test'
# classes: 0 index is reserved for background
CLASSES = [
    'background', 'apple', 'banana', 'orange'
]
NUM_CLASSES = 4
# whether to visualize images after creating the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False
# location to save model and plots

OUT_DIR = '../outputs'
SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # save model after these many epochs