import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import save_image, make_grid

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

import os
import sys
from PIL import Image
import itertools
import time
import datetime
import random
import glob, itertools

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#img_size = 256
#batch_size = 2
#conv_dim = 16
#dataset_name = "Monet"
#g_blocks = 9
#epochs = 16

def img_torch_to_numpy(img):
    img_size = img.size(1)
    return img.detach().cpu().view(3, img_size, img_size).transpose(0, 1).transpose(1, 2).numpy()

def print_Data_images(img_gray, img_color):

    plt.figure(figsize=(30,30))
    
    plt.subplot(141)
    plt.axis('off')
    #plt.set_cmap('Greys')
    plt.imshow(img_torch_to_numpy(img_gray))

    plt.subplot(142)
    plt.axis('off')
    plt.imshow(img_torch_to_numpy(img_color))

    plt.show()

def print_images(img_gray, img_color, img_gen1, img_gen2):
    plt.figure(figsize=(20,20))
    
    plt.subplot(141)
    plt.axis('off')
    plt.title('A')
    plt.imshow(img_torch_to_numpy(img_gray))

    plt.subplot(142)
    plt.axis('off')
    plt.title('B')
    plt.imshow(img_torch_to_numpy(img_color))

    plt.subplot(143)
    plt.axis('off')
    #img_gen = make_grid(img_gen, nrow=1, normalize=True)
    img_gen = img_torch_to_numpy(img_gen1)
    plt.title('Fake A')
    plt.imshow(img_gen)

    plt.subplot(144)
    plt.axis('off')
    img_gen = img_torch_to_numpy(img_gen2)
    plt.title('Fake B')
    plt.imshow(img_gen)

    plt.show()
    
class Losses(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    self.history = []

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum/self.count
    self.history.append(self.avg)

  def plot(self, title=''):
    plt.plot(self.history)
    plt.title(title)
    plt.show()

  def truncate(self, n):
    self.history = self.history[:-(n % len(self.history))]
    