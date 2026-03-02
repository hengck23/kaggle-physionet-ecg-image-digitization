import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import timm
from timm.models.resnet import *
#from timm.models.convnext import *




########################################################################################
# unet decoder

class MyDecoderBlock(nn.Module):
	def __init__(
			self,
			in_channel,
			skip_channel,
			out_channel,
			scale=2,
	):
		super().__init__()
		self.scale=scale
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channel + skip_channel, out_channel, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channel),
			nn.ReLU(inplace=True),
		)
		self.attention1 = nn.Identity()
		self.conv2 = nn.Sequential(
			nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channel),
			nn.ReLU(inplace=True),
		)
		self.attention2 = nn.Identity()

	def forward(self, x, skip=None):
		x = F.interpolate(x, scale_factor=self.scale, mode='nearest')#, align_corners=False) #nearest bilinear
		if skip is not None:
			x = torch.cat([x, skip], dim=1)
			x = self.attention1(x)
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.attention2(x)
		return x

class MyUnetDecoder(nn.Module):
	def __init__(
			self,
			in_channel,
			skip_channel,
			out_channel,
			scale=[2,2,2,2]
	):
		super().__init__()
		self.center = nn.Identity()

		i_channel = [in_channel, ] + out_channel[:-1]
		s_channel = skip_channel
		o_channel = out_channel
		block = [
			MyDecoderBlock(i, s, o, sc)
			for i, s, o, sc in zip(i_channel, s_channel, o_channel, scale)
		]
		self.block = nn.ModuleList(block)

	def forward(self, feature, skip):
		d = self.center(feature)
		decode = []
		for i, block in enumerate(self.block):
			if 0:
				print(i, d.shape, skip[i].shape if skip[i] is not None else 'none')
				print(block.conv1[0])
				print('')
			s = skip[i]
			d = block(d, s)
			decode.append(d)
		last = d
		return last, decode

#---------------------------------------------------
def encode_with_convnext(e, x):
	encode = []# x 256

	x = e.stem(x)  # 64
	x = e.stages[0](x); encode.append(x)  # 64
	x = e.stages[1](x); encode.append(x)  # 32
	x = e.stages[2](x); encode.append(x)  # 16
	x = e.stages[3](x); encode.append(x)  # 8
	return encode

def encode_with_resnet(e, x):
	encode = []# x 256

	x = e.conv1(x)
	x = e.bn1(x)
	x = e.act1(x)
	#x = e.maxpool(x)

	x = e.layer1(x); encode.append(x)  # 128
	x = e.layer2(x); encode.append(x)  # 64
	x = e.layer3(x); encode.append(x)  # 32
	x = e.layer4(x); encode.append(x)  # 16
	return encode

def F_cross_entropy(logits, truth, ignore_index=255):
	truth = truth.long()
	mask = truth != ignore_index
	if mask.sum() == 0:
		print('all pixels are ignored in bce !!!')
		return logits.sum() * 0

	#class_weights = torch.tensor([1.0, 10.0], device=logits.device)

	ce_loss= F.cross_entropy(
	logits, truth, ignore_index=ignore_index)
	return ce_loss


class Net(nn.Module):
	def __init__(self, pretrained=True, cfg=None):
		super(Net, self).__init__()
		self.output_type = ['infer', 'loss']
		self.register_buffer('D', torch.tensor(0))
		self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
		self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

		#arch = 'convnext_tiny.fb_in22k'  # 'convnext_small.fb_in22k'
		# arch = 'resnet34.a3_in1k'
		arch = 'resnet18d.ra4_e3600_r224_in1k'

		encoder_dim = {
			'resnet18d.ra4_e3600_r224_in1k': [64, 128, 256, 512],
			'resnet34.a3_in1k': [64, 128, 256, 512],
			'convnext_tiny_in22k': [96, 192, 384, 768],
			'convnext_tiny.fb_in22k': [96, 192, 384, 768],
			'convnext_small.fb_in22k': [96, 192, 384, 768], #96, 192, 384, 768
			'convnext_base.fb_in22k': [128, 256, 512, 1024], #96, 192, 384, 768
			'resnet50d': [64, 256, 512, 1024, 2048, ],
		}[arch]
		decoder_dim = [256, 128, 64, 32]

		self.encoder = timm.create_model(
			model_name=arch, pretrained=pretrained, in_chans=3, num_classes=0, global_pool=''
		)
		self.decoder = MyUnetDecoder(
			in_channel=encoder_dim[-1],
			skip_channel=encoder_dim[:-1][::-1]+[0],
			out_channel=decoder_dim,
			scale = [2,2,2,2]
		)
		self.marker = nn.Conv2d(decoder_dim[-1], 13 + 1, kernel_size=1)  # lead name text
		self.orientation = nn.Linear(encoder_dim[-1], 8)

	def forward(self, batch):
		device = self.D.device
		image = batch['image'].to(device)

		B, _3_, H, W = image.shape
		x = image.float() / 255
		x = (x - self.mean) / self.std

		# ---------------------------------------
		e = self.encoder
		encode = encode_with_resnet(e, x)
		pooled = F.adaptive_avg_pool2d(encode[-1],1).reshape(B,-1)
		#[print(f'encode_{i}', e.shape) for i,e in enumerate(encode)]

		last, decode = self.decoder(
			feature=encode[-1], skip=encode[:-1][::-1]+[None]
		)
		#[print(f'decode_{i}', e.shape) for i,e in enumerate(decode)]
		#print('last', last.shape)

		marker = self.marker(last)
		orientation = self.orientation(pooled)

		output = {}
		if 'loss' in self.output_type:
			output['marker_loss'] = F_cross_entropy(
				marker, batch['marker'].to(device))
			output['orientation_loss'] = F_cross_entropy(
				orientation, batch['orientation'].to(device))

		# ----
		if 'infer' in self.output_type:
			output['marker'] = torch.softmax(marker,1)
			output['orientation'] = torch.softmax(orientation,1)

		return output



def run_check_net():
	H, W = 960, 1280
	batch_size = 4

	batch = {
		'image' : torch.from_numpy(np.random.randint(0, 256, (batch_size, 3, H, W))).byte(),
		'marker': torch.from_numpy(np.random.choice(3, (batch_size, H, W))).byte(),
		'orientation': torch.from_numpy(np.random.choice(5, (batch_size))).byte(),
	}

	net = Net(pretrained=True).cuda()
	# print(net)

	with torch.no_grad():
		with torch.amp.autocast('cuda'):
			output = net(batch)
	# ---

	print('batch')
	for k, v in batch.items():
		print(f'{k:>32} : {v.shape} ')

	print('output')
	for k, v in output.items():
		if 'loss' not in k:
			print(f'{k:>32} : {v.shape} ')

	print('loss')
	for k, v in output.items():
		if 'loss' in k:
			print(f'{k:>32} : {v.item()} ')

# main #################################################################
if __name__ == '__main__':
	run_check_net()