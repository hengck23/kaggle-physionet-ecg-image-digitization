import sys
sys.path.append('../third_party')
from my_helper import *

from model import *
from dataset import *

import cv2
from timeit import default_timer as timer
import os

#####################################################################################
#configuration
cfg = dotdict(

	lr=0.001,
	train_batch_size=16,
	train_num_worker=32,
	max_num_iteration=16000,
	num_train_id=30, # use only first 300

	#input
	kaggle_dir= \
		'/media/hp/8TB-HDD/work/2025/kaggle/physionet-digitization/data/physionet-ecg-image-digitization',
	# gridpoint data
	processed_dir= \
		'/media/hp/8TB-HDD/work/2025/kaggle/physionet-digitization/data/processed/gridpoint_xy',

	#output
	out_dir = \
		'/media/hp/8TB-HDD/work/2025/kaggle/physionet-digitization/result/final-submit-01/stg0-resnet18d-00',
)
pretrain_file = None #resume from previous training



## start here ############################################
os.makedirs(f'{cfg.out_dir}/checkpoint', exist_ok=True)


net = Net(pretrained=True)
start_iteration = 0
if pretrain_file:
	f = torch.load(
		pretrain_file,
		map_location=lambda storage, loc: storage)
	state_dict = f['state_dict']
	print(net.load_state_dict(state_dict, strict=False))
	start_iteration =f['iteration']

net = net.cuda()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.lr)

# train data
train_df = pd.read_csv(f'{cfg.kaggle_dir}/train.csv')
train_id = train_df['id'].astype(str).values[:cfg.num_train_id]

print('load data ...')
data = load_all_data(
	train_id=train_id,
	kaggle_dir=cfg.kaggle_dir,
	processed_dir=cfg.processed_dir,
)
print('')

train_dataset = ECGDataset(data)
train_loader = DataLoader(
	train_dataset,
	sampler=RandomSampler(train_dataset),
	batch_size=cfg.train_batch_size,
	drop_last=True,
	num_workers=cfg.train_num_worker,
	pin_memory=True,
	worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
	collate_fn=null_collate,
)

#####################################################################
log = Logger()
log.open(f'{cfg.out_dir}/log.train.txt', mode='a')
log.write(f'\n--- [START {log.timestamp()}] {"-" * 64}')
log.write(f'__file__ = {__file__}')
log.write(f'')

net.train()
net.output_type = ['loss', 'infer']

iteration = start_iteration
start_timer = timer()

log.write(f'iteration   loss     time_taken')
log.write(f'-------------------------------')
while(True):
	for t, batch in enumerate(train_loader):
		with torch.amp.autocast('cuda', dtype=torch.bfloat16):  # bfloat16  float32
			output = net(batch)
			loss = output['marker_loss'] + output['orientation_loss']
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# ---
		if iteration % 100 == 0:
			timestamp = time_to_str(timer() - start_timer, 'min')
			log.write(f'{iteration:08d}  {loss.item():2.8f}  {timestamp}')

			if iteration!=start_iteration:
				torch.save({
					'state_dict': net.state_dict(),
					'iteration': iteration,
				}, f'{cfg.out_dir}/checkpoint/{iteration:08d}.checkpoint.pth')
				torch.save({
					'state_dict': net.state_dict(),
					'iteration': iteration,
				}, f'{cfg.out_dir}/checkpoint/last.checkpoint.pth')

			if 1: #optional visulisation
				image = batch['image'].permute(0, 2, 3, 1).contiguous().data.cpu().numpy()
				marker = output['marker'].argmax(1).data.cpu().numpy().astype(np.uint8)

				B = len(image)
				for b in range(B):
					m = image[b]
					h, w = m.shape[:2]
					zeros = np.zeros((h, w), dtype=np.float32)

					gray = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
					gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
					overlay = gray//2
					#overlay = 255 - (255 - overlay / 2) * (1 - g1)
					overlay = overlay.astype(np.uint8)

					mk = marker[b]
					mk = cv2.applyColorMap(mk, MLUT)
					overlay[mk != 0] = mk[mk != 0]
					show_image(m, 'image', cv2.WINDOW_AUTOSIZE)  # WINDOW_NORMAL
					show_image(overlay, 'overlay', cv2.WINDOW_AUTOSIZE)
					cv2.waitKey(1)

		# ---
		iteration += 1
		if iteration >= cfg.max_num_iteration: exit(0)
