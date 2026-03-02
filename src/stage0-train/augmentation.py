import cv2
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')


############################################################################
import albumentations as A

decolor_transform = A.OneOf([
	A.ToGray(p=1.0),  # RGB → Gray (still 3-channel)
	A.ChannelShuffle(p=1.0),  # Random RGB swap
	A.HueSaturationValue(hue_shift_limit=15,
	                     sat_shift_limit=30,
	                     val_shift_limit=15, p=1.0),  # Random hue/saturation
	A.RandomBrightnessContrast(brightness_limit=0.2,
	                           contrast_limit=0.2, p=1.0)  # Random intensity/contrast
], p=1.0)

#############################################################################
from noise import pnoise2

#https://www.gmschroeder.com/blog/intro_pyart1.html
def perlin_noise_2d(
	H, W,
):
	scale = np.random.uniform(30,80)        #size
	octaves    = np.random.randint(3,6)        #4 #detail of noise
	persistence = np.random.uniform(0.3,0.8) #0.5,
	lacunarity  = np.random.uniform(1.5,3)    #2,
	seed = np.random.randint(0, 1000)

	patch = np.zeros((H, W), dtype=np.float32)
	for i in range(H):
		for j in range(W):
			patch[i, j] = pnoise2(
				i / scale, j / scale,
				octaves=octaves,
				persistence=persistence,
				lacunarity=lacunarity,
				repeatx=1024, repeaty=1024,
				base=seed,
			)
	# Normalize to [0,1]
	patch = (patch - patch.min()) / (patch.max() - patch.min())
	return patch


def make_dirt_patch(size):
	noise = perlin_noise_2d(size, size)

	# random nonlinear contrast
	noise = np.power(noise, np.random.uniform(1.5, 3.0))
	# random threshold to isolate blobs
	thresh = np.random.uniform(0.3, 0.6)
	mask = (noise > thresh).astype(np.float32)

	# smooth edges
	k = np.random.randint(3, 15)
	mask = cv2.GaussianBlur(mask, (k | 1, k | 1), 0)
	dirt = 1 - mask * (noise)
	return dirt


