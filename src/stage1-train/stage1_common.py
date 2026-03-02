import torch
import cv2
import numpy as np
import cc3d
import os

from stage1_model import Net as Stage1Net
from timeit import default_timer as timer

from scipy.interpolate import griddata
import torch
import torch.nn.functional as F

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
# cv2.applyColorMap(marker, cv2.COLORMAP_JET)

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
#more visualisation

def draw_keypoint(image, gridpoint_xy):
	gridpoint_xy = np.round(gridpoint_xy).astype(np.int32)

	label=1
	for j, i in [
		[19, 3],
		[26, 3],
		[33, 3],
	]:
		color = [255,255,255]
		x, y = gridpoint_xy[j, i];
		cv2.circle(image, (x, y), 5, color,   2)
		cv2.putText(
			image,
			text=LABEL_TO_LEAD_NAME[label],  # text string
			org=(x, y),  # bottom-left corner (x, y)
			fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # font type
			fontScale=1,  # text size
			color=(255, 255, 255),  # text color (B, G, R)
			thickness=2,  # line thickness
			lineType=cv2.LINE_AA  # anti-aliased line
		)
		label +=1

		x, y = gridpoint_xy[j, i + 13];
		cv2.circle(image, (x, y), 5, color,   2)
		cv2.putText(
			image,
			text=LABEL_TO_LEAD_NAME[label],  # text string
			org=(x, y),  # bottom-left corner (x, y)
			fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # font type
			fontScale=1,  # text size
			color=(255, 255, 255),  # text color (B, G, R)
			thickness=2,  # line thickness
			lineType=cv2.LINE_AA  # anti-aliased line
		)
		label +=1

		x, y = gridpoint_xy[j, i + 25];
		cv2.circle(image, (x, y), 5, color,   2)
		cv2.putText(
			image,
			text=LABEL_TO_LEAD_NAME[label],  # text string
			org=(x, y),  # bottom-left corner (x, y)
			fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # font type
			fontScale=1,  # text size
			color=(255, 255, 255),  # text color (B, G, R)
			thickness=2,  # line thickness
			lineType=cv2.LINE_AA  # anti-aliased line
		)
		label +=1

		x, y = gridpoint_xy[j, i + 38];
		cv2.circle(image, (x, y), 5, color,   2)
		cv2.putText(
			image,
			text=LABEL_TO_LEAD_NAME[label],  # text string
			org=(x, y),  # bottom-left corner (x, y)
			fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # font type
			fontScale=1,  # text size
			color=(255, 255, 255),  # text color (B, G, R)
			thickness=2,  # line thickness
			lineType=cv2.LINE_AA  # anti-aliased line
		)
		label +=1



def draw_mapping(image, gridpoint_xy):
	overlay = image.copy() // 2
	for x, y in gridpoint_xy.reshape(-1, 2):
		# print(x, y)
		y = ROUND(y)
		x = ROUND(x)
		if (x == 0) | (y == 0): continue
		cv2.circle(overlay, (x, y), 2, [0, 255, 0], -1)
	draw_keypoint(overlay, gridpoint_xy)
	return overlay


def draw_results_stage1(more):
	gvfiltered = more['ghfiltered']
	ghfiltered = more['gvfiltered']
	ghfiltered = cv2.applyColorMap(ghfiltered, GLUT)
	gvfiltered = cv2.applyColorMap(gvfiltered, GLUT)
	return ghfiltered, gvfiltered

############################################################################################

def load_net(net, checkpoint_file):
	f = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
	state_dict = f['state_dict']
	print(net.load_state_dict(state_dict, strict=False))

	net.eval()
	net.output_type = ['infer']
	return net


#############################################################################################
#rectify

def rectify_image(
	image, gridpoint_xy
):
	H, W = 1700, 2200 #reference size (output size)

	H1,W1 = image.shape[:2]
	sparse_map = gridpoint_xy/[[[W1-1,H1-1]]]*2 -1
	sparse_map = torch.from_numpy(np.ascontiguousarray(sparse_map.transpose(2, 0, 1))).unsqueeze(0).float()
	dense_map  = F.interpolate(sparse_map, size=(H, W), mode='bilinear', align_corners=True)
	distort    = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0).float()
	rectified = F.grid_sample(
		distort, dense_map.permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=False
	)
	rectified = rectified.data.cpu().numpy()
	rectified = rectified[0].transpose(1, 2, 0).astype(np.uint8)
	return rectified

def interpolate_mapping(gridpoint_xy):
	mx, my = np.meshgrid(np.arange(0, 57), np.arange(0, 44))
	coord = np.stack([mx, my], axis=-1).reshape(-1, 2)
	value = gridpoint_xy.copy()
	value = value.reshape(-1, 2)
	missing = np.all(value == [0, 0], axis=1)
	interpolate_xy = griddata(coord[~missing], value[~missing], (mx, my), method='cubic')
	interpolate_xy[np.isnan(interpolate_xy)] = 0
	return interpolate_xy
###############################################################################
## line processing
def segment_to_endpoints_fitline(mask):
	"""
	mask: uint8 binary image (0/255 or 0/1). Non-zero = the segment.
	returns: ((x1, y1), (x2, y2)) as ints, or None if too few pixels.
	"""
	# collect foreground pixels
	ys, xs = np.nonzero(mask)
	if xs.size < 2:
		return None

	pts = np.column_stack([xs, ys]).astype(np.float32)  # (N,2) as (x,y)

	# Optional: subsample for speed if huge
	if pts.shape[0] > 20000:
		idx = np.random.choice(pts.shape[0], 20000, replace=False)
		pts = pts[idx]

	# Fit an infinite line: returns direction (vx,vy) and a point on the line (x0,y0)
	vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

	# Unit direction
	u = np.array([vx, vy], dtype=np.float32)
	u /= (np.linalg.norm(u) + 1e-12)

	p0 = np.array([x0, y0], dtype=np.float32)

	# Project all points onto the line to find extremal parameters
	t = ((pts - p0) @ u)  # (N,)
	t_min, t_max = t.min(), t.max()

	pA = p0 + t_min * u
	pB = p0 + t_max * u

	# Round to integer pixel coords
	return int(round(pA[0])), int(round(pA[1])), int(round(pB[0])), int(round(pB[1]))


def line_params(x1, y1, x2, y2):
	theta = np.arctan2(y2 - y1, x2 - x1)
	nx, ny = -np.sin(theta), np.cos(theta)
	rho = x1 * nx + y1 * ny
	return theta, rho, (nx, ny)


def compare_segment(seg1, seg2, angle_thr=np.deg2rad(3), rho_thr=10, gap_thr=15):
	"""
	seg1 = main
	seg = (x1, y1, x2, y2)
	thresholds in pixels & radians.
	Returns: score in [0,1] or bool
	"""
	x11, y11, x12, y12 = seg1
	x21, y21, x22, y22 = seg2

	theta1, ρ1, n1 = line_params(x11, y11, x12, y12)
	theta2, ρ2, n2 = line_params(x21, y21, x22, y22)

	dtheta = np.abs((theta1 - theta2 + np.pi / 2) % np.pi - np.pi / 2)  # wrapped diff
	dρ = abs(ρ1 - ρ2)

	#if dtheta > angle_thr or dρ > rho_thr:
	#	pass
	# return 0.0  # clearly different lines

	# project segment endpoints onto the unit direction of seg1
	u = np.array([np.cos(theta1), np.sin(theta1)])
	p1 = np.dot([x11, y11], u)
	p2 = np.dot([x12, y12], u)
	q1 = np.dot([x21, y21], u)
	q2 = np.dot([x22, y22], u)

	# compute overlap/gap along the direction
	# min_gap = max(0, max(p1, p2, q1, q2) - min(p1, p2, q1, q2) - (abs(p2 - p1) + abs(q2 - q1)))
	#score = np.exp(-dth eta * 20) * np.exp(-dρ * 0.1) * np.exp(-min_gap / gap_thr)
	dxy = np.min([
		np.abs(x11-x21) + np.abs(y11-y21),
		np.abs(x11-x22) + np.abs(y11-y22),
		np.abs(x12-x21) + np.abs(y12-y21),
		np.abs(x12-x22) + np.abs(y12-y22),
	])

	dtheta = dtheta/np.pi *180
	return dtheta, dρ, dxy  # score


def canonical_order(x1, y1, x2, y2):
	# enforce increasing projection along main direction
	if (x2 < x1) or (x2 == x1 and y2 < y1):
		x1, y1, x2, y2 = x2, y2, x1, y1
	return x1, y1, x2, y2


def canonical_y_order(x1, y1, x2, y2):
	# enforce increasing projection along main direction
	if (y2 < y1):
		x1, y1, x2, y2 = x2, y2, x1, y1
	return x1, y1, x2, y2

def canonical_x_order(x1, y1, x2, y2):
	# enforce increasing projection along main direction
	if (x2 < x1):
		x1, y1, x2, y2 = x2, y2, x1, y1
	return x1, y1, x2, y2
###############################################################################



def output_to_predict(image, batch, output):
	marker = output['marker'][0]
	gridpoint = output['gridpoint'][0, 0]
	gridhline = output['gridhline'][0]
	gridvline = output['gridvline'][0]

	marker = marker.argmax(0).byte().data.cpu().numpy()
	gridpoint = gridpoint.float().data.cpu().numpy()
	gridhline = gridhline.float().data.cpu().numpy()
	gridvline = gridvline.float().data.cpu().numpy()

	gridline  = np.stack([1 - gridhline[0], 1 - gridvline[0]])
	gridvline = gridvline.argmax(0).astype(np.uint8)
	gridhline = gridhline.argmax(0).astype(np.uint8)


	# =========================================================
	# point
	cc = cc3d.connected_components(gridpoint > 0.5)
	stats = cc3d.statistics(cc)
	point_yx = stats['centroids'][1:]
	##print('point_yx:', len(point_yx))

	point_yx = np.round(point_yx).astype(np.int32)
	#overlay = image.copy() // 2
	#overlay[point_yx[:, 0], point_yx[:, 1]] = [0, 255, 0]

	# =========================================================
	# vline

	# get main segment
	gvfiltered = np.zeros_like(gridvline)
	cc = cc3d.connected_components(gridvline != 0)
	num_line = cc.max()
	for l in range(1, num_line + 1):
		t = (cc == l)
		bincount = np.bincount(gridvline[t])
		c = bincount.argmax()
		gvfiltered[t] = c

	# link to main segment
	gvreject = np.zeros_like(gridvline)
	for l in range(1, num_line + 1):
		cc = cc3d.connected_components(gvfiltered == l)
		if cc.max() > 1:
			# check if fp or discontinous
			num = cc.max() + 1  # num segment
			stats = cc3d.statistics(cc)
			area = stats['voxel_counts'][1:]
			label = np.arange(1, num)
			assert (len(area) == len(label))

			argsort = np.argsort(area)[::-1]
			area = area[argsort]
			label = label[argsort]

			# main_segment
			if area[0] < 7: continue
			main_segment = segment_to_endpoints_fitline(cc == label[0])
			main_segment = canonical_y_order(*main_segment)

			for j in range(1, len(label)):
				if area[j] < 7: continue

				# mask = cc == label[j]
				segment = segment_to_endpoints_fitline(cc == label[j])
				segment = canonical_y_order(*segment)
				ang_dis, ori_dis, seg_dis = compare_segment(main_segment, segment)

				# todo: refine tis afterward.
				# correct method: if distance is short, just compare distance
				# elase, comaore angle and origin dis

				if ori_dis > 5:
					gvreject[cc == label[j]] = 255
				else:
					# print('accept')
					gvfiltered[cc == label[j]] = l
	vcc = gvfiltered.copy()

	# =========================================================
	# hline
	#get main segment
	ghfiltered = np.zeros_like(gridhline)
	cc = cc3d.connected_components(gridhline != 0)
	num_line = cc.max()
	for l in range(1, num_line + 1):
		t = (cc == l)
		bincount = np.bincount(gridhline[t])
		c = bincount.argmax()
		ghfiltered[t] = c

	# link to main segment
	ghreject = np.zeros_like(gridhline)
	for l in range(1, num_line + 1):
		cc = cc3d.connected_components(ghfiltered == l)
		if cc.max() > 1:
			num = cc.max()+1 #num segment
			stats = cc3d.statistics(cc)
			area = stats['voxel_counts'][1:]
			label = np.arange(1, num)
			assert (len(area) == len(label))

			argsort = np.argsort(area)[::-1]
			area = area[argsort]
			label = label[argsort]

			# main_segment
			if area[0] < 7: continue
			main_segment = segment_to_endpoints_fitline(cc == label[0])
			main_segment = canonical_x_order(*main_segment)

			for j in range(1,len(label)):
				if area[j] <7: continue

				segment = segment_to_endpoints_fitline(cc == label[j])
				segment = canonical_x_order(*segment)
				ang_dis, ori_dis, seg_dis = compare_segment(main_segment, segment)

				if ori_dis > 5:
					ghreject[cc == label[j]] = 255
				else:
					# print('accept')
					ghfiltered[cc == label[j]] = l

	hcc = ghfiltered.copy()
	# =========================================================
	# mapping
	gridpoint_xy = np.zeros((44, 57, 2), np.float32)
	for y, x in point_yx:
		uy = ROUND(y)
		ux = ROUND(x)
		j = hcc[uy, ux]
		i = vcc[uy, ux]
		if (j == 0) | (i == 0): continue
		# print(f'({x},{y}) --> ({j},{i})')
		gridpoint_xy[j - 1, i - 1] = [x, y]

	gridpoint_xy = interpolate_mapping(gridpoint_xy)
	more={ #additional results for debug
		'ghfiltered':ghfiltered,
		'gvfiltered':gvfiltered,
	}
	return gridpoint_xy, more

	pass