import torch
import cv2
import numpy as np
import cc3d
import os

from stage0_model import Net as Stage0Net
from timeit import default_timer as timer

##################################################################3
#helper

class dotdict(dict):
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

	def __getattr__(self, name):
		try:
			return self[name]
		except KeyError:
			raise AttributeError(name)

def time_to_str(t, mode='min'):
	if mode=='min':
		t  = int(t/60)
		hr = t//60
		min = t%60
		return '%2d hr %02d min'%(hr,min)

	elif mode=='sec':
		t   = int(t)
		min = t//60
		sec = t%60
		return '%2d min %02d sec'%(min,sec)

	else:
		raise NotImplementedError

def ROUND(x):
	if isinstance(x, list):
		return [int(round(xx)) for xx in x]
	else:
		return int(round(x))

############################################################################################
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
print(THIS_DIR)

HEIGHT, WIDTH = 1152, 1440
LEAD_NAME_TO_LABEL = {
	'None': 0,
	'I': 1,
	'aVR': 2,
	'V1': 3,
	'V4': 4,
	'II': 5,
	'aVL': 6,
	'V2': 7,
	'V5': 8,
	'III': 9,
	'aVF': 10,
	'V3': 11,
	'V6': 12,
	'II-rhythm': 13,
}
LABEL_TO_LEAD_NAME = {
	v: k for k, v in LEAD_NAME_TO_LABEL.items()
}

MLUT = np.zeros((256, 1, 3), dtype=np.uint8)
MLUT[1, 0] = MLUT[5, 0] = MLUT[9, 0] = [255, 255, 255]
MLUT[2, 0] = MLUT[6, 0] = MLUT[10, 0] = [0, 255, 255]
MLUT[3, 0] = MLUT[7, 0] = MLUT[11, 0] = [255, 255, 0]
MLUT[4, 0] = MLUT[8, 0] = MLUT[12, 0] = [255, 0, 255]
MLUT[13, 0] = [255, 255, 255]

#############################################################################################
#visualisation

def show_image(
		image,
		name='image',
		mode=cv2.WINDOW_AUTOSIZE,  # WINDOW_NORMAL
		resize=None
):
	cv2.namedWindow(name, mode)
	if image.ndim == 3:
		cv2.imshow(name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
	if image.ndim == 2:
		cv2.imshow(name, image)
	if resize is not None:
		H, W = image.shape[:2]
		cv2.resizeWindow(name, int(resize * W), int(resize * H))

#############################################################################################
# reference
gridpoint0001_xy = np.load(f'{THIS_DIR}/640106434-0001.gridpoint_xy.npy')
def make_ref_point():
	h0001, w0001 = 1700, 2200
	# lead name
	ref_pt = []
	for j, i in [
		[19, 3],
		[26, 3],
		[33, 3],
	]:
		# x, y = gridpoint0001_xy[j, i];
		# ref_pt.append([x, y])  # cv2.circle(marker, (x, y), MTHICKNESS, color,   -1)
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

REF_PT9 = make_ref_point()
print('REF_PT:', REF_PT9.shape)


def normalise_image(image, pt9, ref_pt9=REF_PT9):
	pt9 = np.array(pt9, np.float32)
	homo, match = cv2.findHomography(pt9, ref_pt9, method=cv2.RANSAC)
	aligned = cv2.warpPerspective(image, homo, (WIDTH, HEIGHT))
	match = match.reshape(-1)
	return aligned, homo, match


############################################################################################

def load_net(net, checkpoint_file):
	f = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
	state_dict = f['state_dict']
	print(net.load_state_dict(state_dict, strict=False))

	net.eval()
	net.output_type = ['infer']
	return net


def image_to_batch(image):
	H, W = image.shape[:2]
	scale = WIDTH / W

	simage = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
	sH, sW = simage.shape[:2]
	pH = int(sH // 32) * 32 + 32
	pW = int(sW // 32) * 32 + 32
	padded = np.pad(simage, [[0, pH - sH], [0, pW - sW], [0, 0]], mode='constant', constant_values=0)

	image0 = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1))).unsqueeze(0)
	batch = {
		'image': [image0],
		'tta': [0]
	}

	for tta in [1, 2, 3]:  # range(1): 1,2,3
		if tta == 1:
			image1 = torch.flip(image0, [2]).contiguous()
		if tta == 2:
			image1 = torch.flip(image0, [3]).contiguous()
		if tta == 3:
			image1 = torch.flip(image0, [2, 3]).contiguous()
		batch['image'].append(image1)
		batch['tta'].append(tta)

	batch['image'] = torch.cat(batch['image'])
	batch['scale'] = scale
	batch['sW'] = sW
	batch['sH'] = sH
	batch['W'] = W
	batch['H'] = H
	return batch


def marker_to_keypoint(image, orientation, marker, scale):
	orientation = orientation.data.cpu().numpy().reshape(-1)
	marker = marker.permute(0, 2, 3, 1).float().data.cpu().numpy()[0]

	k = orientation.argmax()
	# print('orientation', orientation, k)
	if k != 0:
		#print('** unrotated **')
		if k <= 3:
			k = -k
		else:
			print(f'k={k}rotation unknown????')

	marker = np.rot90(marker, k, axes=(0, 1))
	keypoint = []
	thresh = marker.argmax(-1)
	for label in [2, 3, 4, 6, 7, 8, 10, 11, 12]:
		cc = cc3d.connected_components(thresh == label)
		stats = cc3d.statistics(cc)
		center = stats['centroids'][1:]
		area = stats['voxel_counts'][1:]
		argsort = np.argsort(area)[::-1]
		center = center[argsort]
		area = area[argsort]

		#default (missing point)
		center = np.append(center, [[0,0]], axis=0)
		area   = np.append(area, [1], axis=0)

		# choose top 1 only
		for (y, x), a in zip(center[:1], area[:1]):
			leadname = LABEL_TO_LEAD_NAME[label]
			x, y = x / scale, y / scale
			keypoint.append([
				x, y, label, leadname
			])
	return keypoint, k


def output_to_predict(image, batch, output):
	marker = 0
	orientation = 0

	num_tta = len(batch['tta'])
	sH, sW = batch['sH'], batch['sW']
	scale = batch['scale']

	for b in range(num_tta):
		tta = batch['tta'][b]

		mk = output['marker'][[b]]
		on = output['orientation'][b]

		if tta == 1:
			mk = torch.flip(mk, [2]).contiguous()
			on = on[[4, 5, 6, 7, 0, 1, 2, 3]]
		elif tta == 2:
			mk = torch.flip(mk, [3]).contiguous()
			on = on[[6, 7, 4, 5, 2, 3, 0, 1]]
		elif tta == 3:
			mk = torch.flip(mk, [2, 3]).contiguous()
			on = on[[2, 3, 0, 1, 6, 7, 4, 5, ]]
		else:
			pass

		orientation += on
		marker += mk[..., :sH, :sW]

	marker = marker / num_tta
	orientation = orientation / num_tta
	keypoint, k = marker_to_keypoint(image, orientation, marker, scale)
	rotated = np.ascontiguousarray(np.rot90(image, k, axes=(0, 1)))
	return rotated, keypoint,


# ----
# def reprojection_error(H, pt1, pt2):
# 	# pt1: reference points (N×2)
# 	# pt2: query points (N×2)
# 	pt2_xy1 = np.hstack([pt2, np.ones((len(pt2), 1))])
# 	proj_xy1 = (H @ pt2_xy1.T).T
# 	proj = proj_xy1[:, :2] / proj_xy1[:, 2, np.newaxis]
# 	error = np.linalg.norm(proj - pt1, axis=1)
# 	return error

def normalise_by_homography(image, keypoint):
	#[ k[-1] for k in keypoint]
	pt9 = [[ k[0],k[1]] for k in keypoint]
	normalised, homo, match = normalise_image(image, pt9)
	for i in range(len(keypoint)):
		keypoint[i].append(match[i])
	return normalised, keypoint, homo

