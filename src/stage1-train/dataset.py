import sys
sys.path.append('../third_party')
from my_helper import ROUND, show_image, time_to_str

import cv2
import numpy as np
import math
import pandas as pd
from timeit import default_timer as timer
import copy

#import matplotlib.pyplot as plt
#import matplotlib
# matplotlib.use('TkAgg')

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from augmentation import *


###################################################################################3
# marker colormap for applyColorMap(marker, cv2.COLORMAP_JET)

#line color
GLUT = np.zeros((256, 1, 3), dtype=np.uint8)
pattern = np.array([
	[255, 255, 0],
	[255, 0, 255],
	[0, 255, 255],
	[255, 255, 255],
	[255, 0, 0],  # Red
	[0, 255, 0],  # Green
	[0, 0, 255],  # Blue
], dtype=np.uint8)
pattern = np.tile(pattern, (255 // len(pattern) + 1, 1))[:255]  # (256,3)
GLUT[1:] = pattern.reshape(255, 1, 3)

# marker color
MLUT = np.zeros((256, 1, 3), dtype=np.uint8)
MLUT[1, 0] = MLUT[5, 0] = MLUT[9, 0] = [255, 255, 255]
MLUT[2, 0] = MLUT[6, 0] = MLUT[10, 0] = [0, 255, 255]
MLUT[3, 0] = MLUT[7, 0] = MLUT[11, 0] = [255, 255, 0]
MLUT[4, 0] = MLUT[8, 0] = MLUT[12, 0] = [255, 0, 255]
MLUT[13, 0] = [255, 255, 255]

#############################################################

# canonical size of image
HEIGHT,WIDTH = 1152, 1440
CSIZE  = 1024  # crop size

#for annotating ground truth mask
LTHICKNESS = 5
PTHICKNESS = 2
MTHICKNESS = 7

#wrong pseudo label
from error_id_list import ERROR_ID

#reference gridpoint for 0001 image
gridpoint0001_xy = np.load('640106434-0001.gridpoint_xy.npy')

# canonical reference keypoints (for homography normalisation)
def make_ref_point():
	h0001, w0001 = 1700, 2200
	# lead name
	ref_pt = []
	for j, i in [
		[19, 3],
		[26, 3],
		[33, 3],
	]:
		#x, y = gridpoint0001_xy[j, i];
		#ref_pt.append([x, y])  # cv2.circle(marker, (x, y), MTHICKNESS, color,   -1)
		x, y = gridpoint0001_xy[j, i + 13];
		ref_pt.append([x, y])  # cv2.circle(marker, (x, y), MTHICKNESS, color+1, -1)
		x, y = gridpoint0001_xy[j, i + 25];
		ref_pt.append([x, y])  # cv2.circle(marker, (x, y), MTHICKNESS, color+2, -1)
		x, y = gridpoint0001_xy[j, i + 38];
		ref_pt.append([x, y])  # cv2.circle(marker, (x, y), MTHICKNESS, color+3, -1)

	ref_pt = np.array(ref_pt, np.float32)
	scale = 1280 / w0001
	ref_pt = ref_pt * [[scale, scale]]
	shift = (1440 - 1280) / 2
	ref_pt = ref_pt + [[shift, shift]] + [[-6, +10]]
	return ref_pt
REF_PT = make_ref_point()
print('REF_PT:', REF_PT.shape)


#homography normalised image ( gridpoint_xy aligned to REF_PT)
def normalise_image(image, gridpoint_xy):
	pt = []
	for j, i in [
		[19, 3],
		[26, 3],
		[33, 3],
	]:
		#x, y = gridpoint_xy[j, i];
		#pt.append([x, y])  # cv2.circle(marker, (x, y), MTHICKNESS, color,   -1)
		x, y = gridpoint_xy[j, i + 13];
		pt.append([x, y])  # cv2.circle(marker, (x, y), MTHICKNESS, color+1, -1)
		x, y = gridpoint_xy[j, i + 25];
		pt.append([x, y])  # cv2.circle(marker, (x, y), MTHICKNESS, color+2, -1)
		x, y = gridpoint_xy[j, i + 38];
		pt.append([x, y])
	pt = np.array(pt, np.float32)
	homo, mask = cv2.findHomography(pt, REF_PT, method=cv2.RANSAC)
	aligned = cv2.warpPerspective(image, homo, (WIDTH, HEIGHT))

	# Apply H
	Ny, Nx = gridpoint_xy.shape[:2]
	point_xy = gridpoint_xy.reshape(-1, 2)
	invalid = (point_xy[:, 0] == 0) & (point_xy[:, 1] == 0)
	# point_xy = point_xy[~invalid]
	point_xy1 = np.hstack([point_xy, np.ones((point_xy.shape[0], 1))])
	aligned_xy1 = (homo @ point_xy1.T).T
	aligned_xy = aligned_xy1[:, :2] / aligned_xy1[:, 2, np.newaxis]
	aligned_xy[invalid] = 0
	aligned_xy = aligned_xy.reshape(Ny, Nx ,2)

	return aligned, aligned_xy, homo



def load_all_data(
	train_id,
	kaggle_dir,  # original kaggle data
	processed_dir,  # annotated gridpoint data
	norm_dir, #stage0 prediction result
	skip_error_id=ERROR_ID
):

	all_data = []
	n = 0
	start_timer = timer()

	# make input image (homography normalised from annotation )
	for image_id in train_id:
		n = n + 1
		if image_id in skip_error_id: continue

		for type_id in ['0001', '0003', '0004', '0005', '0006', '0009', '0010', '0011', '0012']:
			try:
				# 1. load image
				image = cv2.imread(f'{kaggle_dir}/train/{image_id}/{image_id}-{type_id}.png', cv2.IMREAD_COLOR_RGB)

				# 2. load gridpoint
				if type_id == '0001':
					gridpoint_xy = gridpoint0001_xy.copy()
				else:
					gridpoint_xy = np.load(f'{processed_dir}/{image_id}/{image_id}-{type_id}.gridpoint_xy.npy')

				# 3. make aligned imaged to reference
				image, gridpoint_xy, homo = normalise_image(image, gridpoint_xy)

				timestamp = time_to_str(timer() - start_timer, 'min')
				print(f'\rpreload: {n:3} {timestamp} {image_id}, {type_id}  image', image.shape, end='')

				all_data.append({
					'image_id': image_id,
					'type_id': type_id,
					'image': image,
					'gridpoint_xy': gridpoint_xy,
				})
			except:
				pass
	print('')

	######################################################
	# make input image (homography normalised from stage0 prediction )
	for image_id in train_id:
		n = n + 1
		if image_id in skip_error_id: continue

		#for type_id in ['0001']:
		for type_id in ['0001', '0003', '0004', '0005', '0006', '0009', '0010', '0011', '0012']:

			try:
				# 1. load image
				image = cv2.imread(f'{norm_dir}/{image_id}-{type_id}.norm.png', cv2.IMREAD_COLOR_RGB)

				# 2. load gridpoint
				if type_id == '0001':
					gridpoint_xy = gridpoint0001_xy.copy()
				else:
					gridpoint_xy = np.load(f'{processed_dir}/{image_id}/{image_id}-{type_id}.gridpoint_xy.npy')

				# 3. make aligned imaged to reference using stage0 homography
				# 3.1 load homography matrix H
				homo = np.load(f'{norm_dir}/{image_id}-{type_id}.homo.npy')

				timestamp = time_to_str(timer() - start_timer, 'min')
				print(f'\rpreload(homo): {n:3} {timestamp} {image_id}, {type_id}  image', image.shape, end='')

				# 3.2 apply H
				Ny, Nx = gridpoint_xy.shape[:2]
				point_xy = gridpoint_xy.reshape(-1, 2)
				invalid = (point_xy[:, 0] == 0) & (point_xy[:, 1] == 0)
				# point_xy = point_xy[~invalid]
				point_xy1 = np.hstack([point_xy, np.ones((point_xy.shape[0], 1))])
				aligned_xy1 = (homo @ point_xy1.T).T
				aligned_xy = aligned_xy1[:, :2] / aligned_xy1[:, 2, np.newaxis]
				aligned_xy[invalid] = 0
				aligned_xy = aligned_xy.reshape(Ny, Nx, 2)

				all_data.append({
					'image_id': image_id,
					'type_id': type_id,
					'image': image,
					'gridpoint_xy': aligned_xy,
				})
			except:
				pass
	print('')
	print('all_data:', len(all_data))


	######################################################
	#make ground truth
	for m, data in enumerate(all_data):
		image_id = data['image_id']
		type_id  = data['type_id']
		print(f'mask: \r{m:3} {timestamp} {image_id}, {type_id}  image', image.shape, end='')

		image = data['image']
		gridpoint_xy = data['gridpoint_xy']
		H, W = image.shape[:2]

		# grid line mask
		gridpoint_xy = np.round(gridpoint_xy).astype(np.int32)
		Ny, Nx = gridpoint_xy.shape[:2]
		gpoint = np.zeros((H, W), dtype=np.float32)
		ghline = np.zeros((H, W), dtype=np.uint8)
		gvline = np.zeros((H, W), dtype=np.uint8)
		for x, y in gridpoint_xy.reshape(-1, 2):
			if (x != 0) & (y != 0):
				cv2.circle(gpoint, (x, y), PTHICKNESS, 1, -1)

		for j in range(Ny):
			for i in range(Nx - 1):
				x1, y1 = gridpoint_xy[j, i]
				x2, y2 = gridpoint_xy[j, i + 1]
				if (x1 != 0) & (y1 != 0) & (x2 != 0) & (y2 != 0):
					cv2.line(ghline, (x1, y1), (x2, y2), j+1, LTHICKNESS)  # j

		for j in range(Ny - 1):
			for i in range(Nx):
				x1, y1 = gridpoint_xy[j, i]
				x2, y2 = gridpoint_xy[j + 1, i]
				if (x1 != 0) & (y1 != 0) & (x2 != 0) & (y2 != 0):
					cv2.line(gvline, (x1, y1), (x2, y2), i+1, LTHICKNESS)

		# lead text mask
		marker = np.zeros((H, W), dtype=np.uint8)

		color = 1
		for j, i in [ #3 rows of ECG
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

		if 1: #4th row rhythm strip of ECG
			j, i = 40, 3
			color = 13
			x, y = gridpoint_xy[j,i]; cv2.circle(marker, (x, y), MTHICKNESS, color, -1)

		#update dict
		all_data[m]['gridpoint']=gpoint
		all_data[m]['gridhline']=ghline
		all_data[m]['gridvline']=gvline
		all_data[m]['marker']=marker

		# ----
		if 0: #for debug
			zeros = np.zeros((H, W), dtype=np.uint8)
			gpoint = np.dstack((gpoint, zeros, zeros))

			overlay = image.copy() // 2
			overlay = 255-(255-overlay)*(1-gpoint)

			marker = cv2.applyColorMap(marker, MLUT)
			overlay[marker != 0] = marker[marker != 0]
			overlay = overlay.astype(np.uint8)

			gvline = cv2.applyColorMap(gvline, GLUT)
			ghline = cv2.applyColorMap(ghline, GLUT)
			gline = gvline//2+ghline//2


			show_image(overlay, 'overlay', cv2.WINDOW_NORMAL)  # WINDOW_AUTOSIZE
			show_image(image, 'image', cv2.WINDOW_NORMAL)
			show_image(gline, 'gline', cv2.WINDOW_NORMAL)
			#show_image(marker, 'marker', cv2.WINDOW_NORMAL)
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
	data['gridpoint'] = data['gridpoint'][y0:y1, x0:x1]
	data['gridhline'] = data['gridhline'][y0:y1, x0:x1]
	data['gridvline'] = data['gridvline'][y0:y1, x0:x1]
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

	scale = 1 + np.random.uniform(-0.1, 0.1)
	angle = np.random.uniform(-10, 10) / 180 * math.pi
	rot = np.array([
		[math.cos(angle), -math.sin(angle)],
		[math.sin(angle), math.cos(angle)],
	], dtype=np.float32)

	p = crop_coord.copy()
	p = p - [[CSIZE // 2, CSIZE // 2]]
	p = scale * p
	p = p @ rot.T
	p = p + [[CSIZE // 2, CSIZE // 2]]
	px0, py0 = 0, 0

	if 1:  #
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
	gridpoint = cv2.warpPerspective(
		data['gridpoint'], mat, (CSIZE, CSIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
	gridhline = cv2.warpPerspective(
		data['gridhline'], mat, (CSIZE, CSIZE), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
	gridvline = cv2.warpPerspective(
		data['gridvline'], mat, (CSIZE, CSIZE), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

	# debug
	if 0:
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
	data['gridpoint'] = gridpoint
	data['gridhline'] = gridhline
	data['gridvline'] = gridvline
	return data

def do_full_perspective_transform(data):

	image = data['image']
	H, W = image.shape[:2]
	crop_coord = np.array([
		[0, 0],
		[W - 1, 0],
		[W - 1, H - 1],
		[0, H - 1],
	], dtype=np.float32)

	scale = 1 + np.random.uniform(-0.1, 0.1)
	angle = np.random.uniform(-10, 10) / 180 * math.pi
	rot = np.array([
		[math.cos(angle), -math.sin(angle)],
		[math.sin(angle), math.cos(angle)],
	], dtype=np.float32)

	p = crop_coord.copy()
	p = p - [[W // 2, H // 2]]
	p = scale * p
	p = p @ rot.T
	p = p + [[W // 2, H // 2]]
	px0, py0 = 0, 0

	if 1:  #
		px0 = np.random.uniform(0, int(0.1*W))
		py0 = np.random.uniform(0, int(0.1*H))
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
		image, mat, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=[0, 0, 0])
	marker = cv2.warpPerspective(
		data['marker'], mat, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
	gridpoint = cv2.warpPerspective(
		data['gridpoint'], mat, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
	gridhline = cv2.warpPerspective(
		data['gridhline'], mat, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
	gridvline = cv2.warpPerspective(
		data['gridvline'], mat, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

	# debug
	if 0:
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
	data['gridpoint'] = gridpoint
	data['gridhline'] = gridhline
	data['gridvline'] = gridvline
	return data

def do_full_nudge_transform(data):
	# CS = 320  # 256 #320

	image = data['image']
	H, W = image.shape[:2]
	crop_coord = np.array([
		[0, 0],
		[W - 1, 0],
		[W - 1, H - 1],
		[0, H - 1],
	], dtype=np.float32)

	noise = np.random.uniform(-5, 5,(4,2))
	p = crop_coord+noise

	mat = cv2.getPerspectiveTransform(p.astype(np.float32), crop_coord.astype(np.float32))

	image = cv2.warpPerspective(
		image, mat, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=[0, 0, 0])
	marker = cv2.warpPerspective(
		data['marker'], mat, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
	gridpoint = cv2.warpPerspective(
		data['gridpoint'], mat, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
	gridhline = cv2.warpPerspective(
		data['gridhline'], mat, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
	gridvline = cv2.warpPerspective(
		data['gridvline'], mat, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

	# debug
	if 0:
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
	data['gridpoint'] = gridpoint
	data['gridhline'] = gridhline
	data['gridvline'] = gridvline
	return data

def do_decolor_transform(data):
	image = data['image']
	augmented = decolor_transform(image=image)['image']
	data['image'] = augmented
	return data

def do_motion_blur_transform(data):
	image = data['image']
	u = np.random.choice(3, p=[0.5,0.25,0.25])
	if u==0:
		augmented=image
	if u==1:
		ksize = np.random.choice([5,7,9,11,13,15,17,19])
		augmented = cv2.GaussianBlur(
			image,
			ksize=(ksize, ksize),  # must be odd, e.g. (3,3), (5,5), (15,15)
			sigmaX=0  # 0 → OpenCV computes sigma from kernel size
		)
	if u==2:
		augmented = do_motion_blur(image)
	data['image'] = augmented
	return data

def do_noise_transform(data):
	if np.random.rand() < 0.8:  # 0.8:
		augmented = data['image']
		H,W = augmented.shape[:2]
		size = min(H,W)

		size = int(np.random.uniform(0.5 * 0.8) * size)
		patch = make_dirt_patch(size)
		patch = patch[..., np.newaxis]

		x = np.random.randint(0, W - size + 1)
		y = np.random.randint(0, H - size + 1)

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
			patch = cv2.resize(patch, (W, H))
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
			data = do_full_nudge_transform(data)
		else:
			data = do_full_perspective_transform(data)

		data = do_decolor_transform(data)
		data = do_noise_transform(data)
		data = do_motion_blur_transform(data)

		r = {
			'image': torch.from_numpy(np.ascontiguousarray(data['image'].transpose(2, 0, 1))),
			'marker': torch.from_numpy(data['marker']),
			'gridpoint': torch.from_numpy(data['gridpoint']),
			'gridhline': torch.from_numpy(data['gridhline']),
			'gridvline': torch.from_numpy(data['gridvline']),
		}
		return r


def null_collate(batch):
	d = {}
	key = batch[0].keys()
	for k in key:
		d[k] = [b[k] for b in batch]

	d['image'] = torch.stack(d['image']).byte()
	d['marker'] = torch.stack(d['marker']).byte()
	d['gridpoint'] = torch.stack(d['gridpoint']).float().unsqueeze(1)
	d['gridhline'] = torch.stack(d['gridhline']).byte()
	d['gridvline'] = torch.stack(d['gridvline']).byte()
	return d


# main #################################################################
if __name__ == '__main__':
	#check dataset

	KAGGLE_DIR = \
		'/media/hp/8TB-HDD/work/2025/kaggle/physionet-digitization/data/physionet-ecg-image-digitization'
	PROCESSED_DIR = \
		'/media/hp/8TB-HDD/work/2025/kaggle/physionet-digitization/data/processed/gridpoint_xy'

	norm_dir = \
		'/media/hp/8TB-HDD/work/2025/kaggle/physionet-digitization/result/final-submit-01/stg0-resnet18d-00/normalised-image-from_stage0'

	train_df = pd.read_csv(f'{KAGGLE_DIR}/train.csv')
	train_id = train_df['id'].astype(str).values[:5] #up to first 500 are used

	data = load_all_data(
		train_id=train_id,
		kaggle_dir=KAGGLE_DIR,
		processed_dir=PROCESSED_DIR,
		norm_dir=norm_dir,
	)

	dataset = ECGDataset(data)
	for k in range(len(dataset)):
		#k=0  #fix k to check augmentation

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
		marker = cv2.applyColorMap(marker, GLUT)
		gpoint = data['gridpoint'].data.cpu().numpy()
		ghline = data['gridhline'].data.cpu().numpy()
		gvline = data['gridvline'].data.cpu().numpy()

		zeros = np.zeros_like(gpoint, dtype=np.uint8)
		gpoint = np.dstack((gpoint, zeros, zeros))

		gvline = cv2.applyColorMap(gvline, GLUT)
		ghline = cv2.applyColorMap(ghline, GLUT)
		gline = gvline // 2 + ghline // 2

		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
		overlay = gray//2
		overlay = 255 - (255 - overlay) * (1 - gpoint)
		overlay[marker != 0] = marker[marker != 0]
		overlay = overlay.astype(np.uint8)

		# show_image(image1, 'image1', cv2.WINDOW_AUTOSIZE)
		show_image(overlay, 'overlay,', cv2.WINDOW_AUTOSIZE)
		show_image(image, 'image,', cv2.WINDOW_AUTOSIZE)
		show_image(gline, 'gline', cv2.WINDOW_AUTOSIZE)
		cv2.waitKey(0)
