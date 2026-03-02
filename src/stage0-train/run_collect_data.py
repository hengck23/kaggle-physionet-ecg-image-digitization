import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import pandas as pd

from error_id_list import ERROR_ID
from stage0_model import Net as Stage0Net
from stage0_common import *

DEVICE = 'cuda'
cfg = dotdict(
	float_type=torch.bfloat16,
	num_train_id=20,  # 500, num of train id to predict

	#input
	checkpoint_file=\
		'/media/hp/8TB-HDD/work/2025/kaggle/physionet-digitization/result/final-submit-01/stg0-resnet18d-00/stage0-last.checkpoint.pth',
	kaggle_dir= \
		'/media/hp/8TB-HDD/work/2025/kaggle/physionet-digitization/data/physionet-ecg-image-digitization',

	#output
	out_dir =\
		'/media/hp/8TB-HDD/work/2025/kaggle/physionet-digitization/result/final-submit-01/stg0-resnet18d-00/normalised-image-from_stage0'
)
os.makedirs(cfg.out_dir, exist_ok=True)

########################################################
stage0_net = Stage0Net(pretrained=False)
stage0_net = load_net(stage0_net, cfg.checkpoint_file)
stage0_net.to(DEVICE)

train_df = pd.read_csv(f'{cfg.kaggle_dir}/train.csv')
train_id = train_df['id'].astype(str).values[:cfg.num_train_id]

start_timer = timer()
for n, image_id in enumerate(train_id):
	print(n, image_id, '----------------')
	for type_id in ['0001', '0003','0004', '0005', '0006', '0009', '0010', '0011', '0012']:

		print('\t', type_id)
		image = cv2.imread(f'{cfg.kaggle_dir}/train/{image_id}/{image_id}-{type_id}.png', cv2.IMREAD_COLOR_RGB)
		batch = image_to_batch(image)

		with torch.amp.autocast('cuda', dtype=cfg.float_type):
			with torch.no_grad():
				output = stage0_net(batch)

		rotated, keypoint = output_to_predict(image, batch, output)
		normalised, keypoint, homo = normalise_by_homography(rotated, keypoint)

		#---
		cv2.imwrite(f'{cfg.out_dir}/{image_id}-{type_id}.norm.png', cv2.cvtColor(normalised, cv2.COLOR_RGB2BGR))
		np.save(f'{cfg.out_dir}/{image_id}-{type_id}.homo.npy',homo)

		#optional: show results
		if n<10:
			overlay=rotated//2
			for x, y, label, leadname, match in keypoint:
				x = ROUND(x)
				y = ROUND(y)
				color = tuple(map(int, MLUT[label,0]))

				if match:
					cv2.circle(overlay, (x, y), 10, color, -1)
					cv2.putText(
						overlay,
						text=leadname,  # text string
						org=(x, y),  # bottom-left corner (x, y)
						fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # font type
						fontScale=2,  # text size
						color=(255, 255, 255),  # text color (B, G, R)
						thickness=2,  # line thickness
						lineType=cv2.LINE_AA  # anti-aliased line
					)
				else:
					cv2.circle(overlay, (x,y), 10, color, 2)


			show_image(normalised, 'normalised', cv2.WINDOW_NORMAL, resize=0.5)
			show_image(image, 'image', cv2.WINDOW_NORMAL, resize=0.25)
			show_image(overlay, 'overlay', cv2.WINDOW_NORMAL, resize=0.25)
			cv2.waitKey(1)

	timestamp = time_to_str(timer() - start_timer, 'sec')
	print(timestamp)