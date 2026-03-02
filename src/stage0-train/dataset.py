import sys
sys.path.append('../third_party')
from my_helper import ROUND, show_image, time_to_str

import cv2
import numpy as np
import math
import pandas as pd
from timeit import default_timer as timer
import copy

# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from augmentation import *

###################################################################################

# marker colormap for applyColorMap(marker, cv2.COLORMAP_JET)
MLUT = np.zeros((256, 1, 3), dtype=np.uint8)
MLUT[1, 0] = MLUT[5, 0] = MLUT[9, 0] = [255, 255, 255]
MLUT[2, 0] = MLUT[6, 0] = MLUT[10, 0] = [0, 255, 255]
MLUT[3, 0] = MLUT[7, 0] = MLUT[11, 0] = [255, 255, 0]
MLUT[4, 0] = MLUT[8, 0] = MLUT[12, 0] = [255, 0, 255]
MLUT[13, 0] = [255, 255, 255]


# canonical size of image
#HEIGHT, WIDTH = 960, 1280  # 1440 1080
HEIGHT, WIDTH = 1080, 1440
CSIZE = 1024  # crop size

#for annotating ground truth mask
LTHICKNESS = 5
PTHICKNESS = 2
MTHICKNESS = 7

#wrong pseudo label
from error_id_list import ERROR_ID

#reference gridpoint for 0001 image
gridpoint0001_xy = np.load('640106434-0001.gridpoint_xy.npy')


def load_all_data(
	train_id,
	kaggle_dir,    #original kaggle data
	processed_dir, #annotated gridpoint data
	skip_error_id = ERROR_ID
):
	all_data = []
	n = 0
	start_timer = timer()
	for image_id in train_id:
		n = n + 1
		if image_id in skip_error_id: continue

		for type_id in ['0003', '0001', '0004',   '0005', '0006', '0009',   '0010', '0011', '0012']:
			#1. load image
			image = cv2.imread(f'{kaggle_dir}/train/{image_id}/{image_id}-{type_id}.png', cv2.IMREAD_COLOR_RGB)

			timestamp = time_to_str(timer() - start_timer, 'min')
			print(f'\r{n:3} {timestamp} {image_id}, {type_id}  image', image.shape, end='')

			height, width = image.shape[:2]
			scale = WIDTH / width
			image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
			H, W = image.shape[:2]

			#2. load gridpoint
			if type_id == '0001':
				gridpoint_xy = gridpoint0001_xy.copy()
			else:
				gridpoint_xy = np.load(f'{processed_dir}/{image_id}/{image_id}-{type_id}.gridpoint_xy.npy')

			gridpoint_xy = gridpoint_xy * [[[scale, scale]]]
			gridpoint_xy = gridpoint_xy.astype(np.int32)
			Ny, Nx = gridpoint_xy.shape[:2]

			#3. make ground truth mask
			marker = np.zeros((H, W), dtype=np.uint8)

			# mark lead name
			color = 1
			for j, i in [
				[19, 3],
				[26, 3],
				[33, 3],
			]:
				x, y = gridpoint_xy[j, i];
				cv2.circle(marker, (x, y), MTHICKNESS, color, -1)
				x, y = gridpoint_xy[j, i + 13];
				cv2.circle(marker, (x, y), MTHICKNESS, color + 1, -1)
				x, y = gridpoint_xy[j, i + 25];
				cv2.circle(marker, (x, y), MTHICKNESS, color + 2, -1)
				x, y = gridpoint_xy[j, i + 38];
				cv2.circle(marker, (x, y), MTHICKNESS, color + 3, -1)
				color = color + 4

			# j, i = 40, 3
			# x, y = gridpoint_xy[j,i]; cv2.circle(marker, (x, y), MTHICKNESS, 13, -1)

			all_data.append({
				'image_id': image_id,
				'type_id': type_id,
				'image': image,
				'marker': marker,
				'orientation': 0,
			})

			# ----
			if 0: #for debug
				overlay = image.copy() // 2
				marker_color = cv2.applyColorMap(marker, MLUT)
				overlay[marker_color != 0] = marker_color[marker_color != 0]

				show_image(overlay, 'overlay', cv2.WINDOW_NORMAL)  # WINDOW_AUTOSIZE
				show_image(image, 'image', cv2.WINDOW_NORMAL)
				show_image(marker_color, 'marker_color', cv2.WINDOW_NORMAL)
				cv2.waitKey(0)

	return all_data


############################################################################################
#augmentation
def do_random_crop(data):
	image = data['image']
	H, W = image.shape[:2]

	x0 = np.random.randint(0, W - CSIZE + 1)
	y0 = np.random.randint(0, H - CSIZE + 1)
	x1 = x0 + CSIZE
	y1 = y0 + CSIZE
	data['image'] = image[y0:y1, x0:x1]
	data['marker'] = data['marker'][y0:y1, x0:x1]
	return data


def do_perspective_transform(data):
	image = data['image']
	H, W = image.shape[:2]
	crop_coord = np.array([
		[0, 0],
		[CSIZE - 1, 0],
		[CSIZE - 1, CSIZE - 1],
		[0, CSIZE - 1],
	], dtype=np.float32)

	scale =  np.random.uniform(0.5, 1.1) #1 + np.random.uniform(-0.1, 0.1)
	angle = np.random.uniform(-15, 15) / 180 * math.pi
	rot = np.array([
		[math.cos(angle), -math.sin(angle)],
		[math.sin(angle), math.cos(angle)],
	], dtype=np.float32)

	p = crop_coord.copy()
	p = p - [[CSIZE // 2, CSIZE // 2]]
	p = scale * p
	p = p @ rot.T
	p = p + [[CSIZE // 2, CSIZE // 2]]

	if 1:
		px0 = np.random.uniform(0, W - CSIZE + 1)
		py0 = np.random.uniform(0, H - CSIZE + 1)
		p = p + [[px0, py0]]

	# for x, y in p:
	# 	#print(x, y)
	# 	y = ROUND(y)
	# 	x = ROUND(x)
	# 	cv2.circle(image, (x, y), 3, [255, 0, 0], -1)
	# show_image(image, 'augment point', cv2.WINDOW_NORMAL)
	# cv2.waitKey(0)

	mat = cv2.getPerspectiveTransform(p.astype(np.float32), crop_coord.astype(np.float32))

	image = cv2.warpPerspective(
		image, mat, (CSIZE, CSIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=[0, 0, 0])
	marker = cv2.warpPerspective(
		data['marker'], mat, (CSIZE, CSIZE), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

	if 0:  # debug --------------------------------------------------------
		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
		overlay = gray
		overlay[gpoint_valid > 0.5] = overlay[gpoint_valid > 0.5] // 4
		overlay[gpoint > 0.5] = [[[255, 0, 0]]]

		# show_image(image1, 'image1', cv2.WINDOW_AUTOSIZE)
		show_image(overlay, 'overlay,', cv2.WINDOW_AUTOSIZE)
		show_image(image, 'image,', cv2.WINDOW_AUTOSIZE)
		show_image(gpoint, 'gpoint', cv2.WINDOW_AUTOSIZE)
		show_image(gpoint_valid, 'gpoint_valid', cv2.WINDOW_AUTOSIZE)
		cv2.waitKey(0)

	# todo: update gpoint_xy
	data['image'] = image
	data['marker'] = marker
	return data


def do_flip_transform(data):

	image = data['image']
	marker = data['marker']

	k = np.random.choice(8)
	rot, flipud = [
		[0, False],
		[1, False],
		[2, False],
		[3, False],
		[0, True],
		[1, True],
		[2, True],
		[3, True],
	][k]

	if rot != 0:
		image = np.rot90(image, rot, axes=(0, 1))
		marker = np.rot90(marker, rot, axes=(0, 1))

	if flipud:
		image = image[::-1]
		marker = marker[::-1]

	# swap if train patch is small
	# if rot in [1, 3]:
	# 	# 90° or 270° — swap horizontal/vertical
	# 	ghline = grid[...,1].copy()
	# 	gvline = grid[...,2].copy()
	# 	grid[...,1] = gvline
	# 	grid[...,2] = ghline

	data['image'] = np.ascontiguousarray(image)
	data['marker'] = np.ascontiguousarray(marker)
	data['orientation'] = k
	return data


def do_decolor_transform(data):
	image = data['image']
	augmented = decolor_transform(image=image)['image']
	data['image'] = augmented
	return data

def do_noise_transform(data):
	if np.random.rand() < 0.8:  # 0.8:
		augmented = data['image']

		size = int(np.random.uniform(0.3, 0.8) * CSIZE)
		patch = make_dirt_patch(size)
		patch = patch[..., np.newaxis]

		x = np.random.randint(0, CSIZE - size + 1)
		y = np.random.randint(0, CSIZE - size + 1)

		u = np.random.choice([1, 2, 3, 4], p=[0.50, 0.20, 0.20, 0.10])
		# u = np.random.choice([1, 2, 3, 4], p=[0, 0, 0, 1])

		if u == 1:  # multiple
			augmented[y:y + size, x:x + size] = augmented[y:y + size, x:x + size] * patch
		if u == 2:  # screen
			augmented[y:y + size, x:x + size] = 255 - (255 - augmented[y:y + size, x:x + size]) * patch
		if u == 3:  # mixed
			augmented[y:y + size, x:x + size] = augmented[y:y + size, x:x + size] * patch
			augmented[y:y + size, x:x + size] = 255 - (255 - augmented[y:y + size, x:x + size]) * patch
		if u == 4:  # hole
			patch = cv2.resize(patch, (CSIZE, CSIZE))
			c = np.random.choice(255)
			augmented[patch < 0.8] = [[c, c, c]]

		data['image'] = augmented
	return data

class ECGDataset(Dataset):
	def __init__(self, data):
		self.data = data
		self.length = len(self.data)

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		i = index
		data = copy.deepcopy(self.data[i])

		if np.random.rand() < 0.1:
			data = do_random_crop(data)
		else:
			data = do_perspective_transform(data)

		data = do_flip_transform(data)
		data = do_decolor_transform(data)
		data = do_noise_transform(data)

		r = {
			'image': torch.from_numpy(np.ascontiguousarray(data['image'].transpose(2, 0, 1))),
			'marker': torch.from_numpy(data['marker']),
			'orientation': data['orientation'],
		}
		return r

def null_collate(batch):
	d = {}
	key = batch[0].keys()
	for k in key:
		d[k] = [b[k] for b in batch]

	d['image'] = torch.stack(d['image']).byte()
	d['marker'] = torch.stack(d['marker']).byte()
	d['orientation'] = torch.tensor(d['orientation']).long()
	return d


# main #################################################################
if __name__ == '__main__':

	#check dataset

	KAGGLE_DIR = \
		'/media/hp/8TB-HDD/work/2025/kaggle/physionet-digitization/data/physionet-ecg-image-digitization'
	PROCESSED_DIR = \
		'/media/hp/8TB-HDD/work/2025/kaggle/physionet-digitization/data/processed/gridpoint_xy'

	train_df = pd.read_csv(f'{KAGGLE_DIR}/train.csv')
	train_id = train_df['id'].astype(str).values[:5]

	data = load_all_data(
		train_id=train_id,
		kaggle_dir=KAGGLE_DIR,
		processed_dir=PROCESSED_DIR,
	)
	dataset = ECGDataset(data)
	for k in range(len(dataset)):
		data = dataset[k]

		image = data['image'].data.cpu().numpy()
		image = np.ascontiguousarray(image.transpose(1, 2, 0))


		if 0:
			augmented = image.copy()
			size = 256
			patch = make_dirt_patch(size)
			x = (CS - size) // 2 - 1
			y = (CS - size) // 2 - 1
			mask = np.ones((CS, CS, 3), np.float32)
			mask[y:y + size, x:x + size] = patch[..., np.newaxis]

			augmented = image
			augmented = 255 - (255 - augmented) * (mask)  # screen
			augmented = augmented * mask  # multiple
			# show_image(np.hstack([image,mask*255,augmented]).astype(np.uint8), 'augmented,', cv2.WINDOW_NORMAL)
			# cv2.waitKey(0)


		marker = data['marker'].data.cpu().numpy().astype(np.uint8)
		marker = cv2.applyColorMap(marker, MLUT)
		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
		overlay = gray
		overlay[marker != 0] = marker[marker != 0]

		# show_image(image1, 'image1', cv2.WINDOW_AUTOSIZE)
		show_image(overlay, 'overlay,', cv2.WINDOW_AUTOSIZE)
		show_image(image, 'image,', cv2.WINDOW_AUTOSIZE)
		cv2.waitKey(0)








