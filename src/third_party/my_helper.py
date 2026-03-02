import cv2
import numpy as np
import pandas as pd

#############################################################################################
# configuration class
class dotdict(dict):
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

	def __getattr__(self, name):
		try:
			return self[name]
		except KeyError:
			raise AttributeError(name)

#############################################################################################
# etc
def ROUND(x):
	if isinstance(x, list):
		return [int(round(xx)) for xx in x]
	else:
		return int(round(x))

def short_e_format(s):
	s = s.replace('e-0', 'e-').replace('e+0','e+')
	return s

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

def np_float32_to_uint8(x, scale=255):
	return (x*scale).astype(np.uint8)

def np_uint8_to_float32(x, scale=255):
	return (x/scale).astype(np.float32)

def int_tuple(x):
	return tuple( [int(round(xx)) for xx in x] )

#############################################################################################
#opencv visualisation
def show_image(
	image,
	name='image',
	mode=cv2.WINDOW_AUTOSIZE,  #WINDOW_NORMAL
	resize=None
):
	cv2.namedWindow(name, mode)
	if image.ndim == 3:
		cv2.imshow(name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
	if image.ndim == 2:
		cv2.imshow(name, image)
	if resize is not None:
		H, W = image.shape[:2]
		cv2.resizeWindow(name, int(resize*W), int(resize*H))


def image_show(name, image, type=None, resize=1):
    if type == 'rgb2bgr':
        image = np.ascontiguousarray(image[:, :, ::-1])

    H, W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_GUI_NORMAL)  # WINDOW_NORMAL
    # cv2.namedWindow(name, cv2.WINDOW_GUI_EXPANDED)  #WINDOW_GUI_EXPANDED
    cv2.imshow(name, image)
    cv2.resizeWindow(name, round(resize * W), round(resize * H))


#############################################################################################
# logging file helper

#from lib.include import *
import os
import pickle
import sys
import pandas as pd
import shutil
import copy
from datetime import datetime

import builtins
import re

class Struct(object):
	def __init__(self, is_copy=False, **kwargs):
		self.add(is_copy, **kwargs)

	def add(self, is_copy=False, **kwargs):
		#self.__dict__.update(kwargs)

		if is_copy == False:
			for key, value in kwargs.items():
				setattr(self, key, value)
		else:
			for key, value in kwargs.items():
				try:
					setattr(self, key, copy.deepcopy(value))
					#setattr(self, key, value.copy())
				except Exception:
					setattr(self, key, value)

	def drop(self,  missing=None, **kwargs):
		drop_value = []
		for key, value in kwargs.items():
			try:
				delattr(self, key)
				drop_value.append(value)
			except:
				drop_value.append(missing)
		return drop_value

	def __str__(self):
		text =''
		for k,v in self.__dict__.items():
			text += '\t%s : %s\n'%(k, str(v))
		return text


def remove_comments(lines, token='#'):
	""" Generator. Strips comments and whitespace from input lines.
	"""

	l = []
	for line in lines:
		s = line.split(token, 1)[0].strip()
		if s != '':
			l.append(s)
	return l

# def open(file, mode=None, encoding=None):
#     if mode == None: mode = 'r'
#
#     if '/' in file:
#         if 'w' or 'a' in mode:
#             dir = os.path.dirname(file)
#             if not os.path.isdir(dir):  os.makedirs(dir)
#
#     f = builtins.open(file, mode=mode, encoding=encoding)
#     return f

def remove(file):
	if os.path.exists(file): os.remove(file)

def empty(dir):
	if os.path.isdir(dir):
		shutil.rmtree(dir, ignore_errors=True)
	else:
		os.makedirs(dir)

# http://stackoverflow.com/questions/34950201/pycharm-print-end-r-statement-not-working
class Logger(object):
	def timestamp(self):
		h = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
		return h

	def __init__(self):
		self.terminal = sys.stdout  #stdout
		self.file = None

	def open(self, file, mode=None):
		if mode is None: mode ='w'
		self.file = open(file, mode)

	def write(self, message, is_terminal=1, is_file=1, end='\n' ):
		if '\r' in message: is_file=0

		if is_terminal == 1:
			self.terminal.write(message+end)
			self.terminal.flush()
			#time.sleep(1)

		if is_file == 1:
			self.file.write(message+end)
			self.file.flush()

	def flush(self):
		# this flush method is needed for python 3 compatibility.
		# this handles the flush command by doing nothing.
		# you might want to specify some extra behavior here.
		pass


