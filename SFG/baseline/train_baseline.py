#%% import packages
import os
import sys
import subprocess
GPU_ID = subprocess.getoutput('nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1| head -n 1 | xargs')
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

import argparse
import pdb
from timeit import default_timer
import yaml
import hashlib
import socket
import random
# ======== PLEASE MODIFY ========
# where is the repo
repoRoot = r'.'

os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
# Flying Chairs Dataset
chairs_path = r'F:\linge\data2\FlyingChairs\FlyingChairs_release\data'
chairs_split_file = r'F:\linge\data2\FlyingChairs\FlyingChairs_release\FlyingChairs_train_val.txt'

import numpy as np
import mxnet as mx

# data readers
from reader.chairs import binary_reader, trainval, ppm, flo
from reader import sintel, kitti, hd1k, things3d
from reader.ANHIR import LoadANHIR, ANHIRPredict
import cv2

#%% parse arguments

model_parser = argparse.ArgumentParser(add_help=False)
training_parser = argparse.ArgumentParser(add_help=False)
training_parser.add_argument('--batch', type=int, default=8, help='minibatch size of samples per device')

parser = argparse.ArgumentParser(parents=[model_parser, training_parser])

parser.add_argument('config', type=str, nargs='?', default=None)
parser.add_argument('--dataset_cfg', type=str, default='chairs.yaml')
parser.add_argument('--relative', type=str, default="")

parser.add_argument('-s', '--shard', type=int, default=1, help='')
parser.add_argument('-w', '--weight', type=int, default=200, help='dist weight when training')

parser.add_argument('-g', '--gpu_device', type=str, default='0', help='Specify gpu device(s)')
parser.add_argument('-c', '--checkpoint', type=str, default=None,
	help='model checkpoint to load; by default, the latest one.'
	'You can use checkpoint:steps to load to a specific steps')
parser.add_argument('--clear_steps', action='store_true')

parser.add_argument('-n', '--network', type=str, default='MaskFlownet') # gl, here only MastFlownet can be the input, contains Flownset_S and others.

parser.add_argument('--debug', action='store_true', help='Do debug')
parser.add_argument('--valid', action='store_true', help='Do validation')
parser.add_argument('--predict', action='store_true', help='Do prediction')
parser.add_argument('--predict_fold', type=str, choices=['train', 'val', 'test'], default='')
parser.add_argument('--visualize', action='store_true', help='Do visualization')

parser.add_argument('--resize', type=str, default='')
parser.add_argument('--prep', type=str, default=None)

args = parser.parse_args()
ctx = [mx.cpu()] if args.gpu_device == '' else [mx.gpu(gpu_id) for gpu_id in map(int, args.gpu_device.split(','))]
infer_resize = [int(s) for s in args.resize.split(',')] if args.resize else None

#%%
## configure networks, data loaders, and logging

import network.config
# load network configuration
with open(os.path.join(repoRoot, 'network', 'config', args.config)) as f:
	config =  network.config.Reader(yaml.safe_load(f))
# load training configuration
with open(os.path.join(repoRoot, 'network', 'config', args.dataset_cfg)) as f:
	dataset_cfg = network.config.Reader(yaml.safe_load(f))
validation_steps = dataset_cfg.validation_steps.value
checkpoint_steps = dataset_cfg.checkpoint_steps.value
max_steps = config.optimizer.learning_rate.value[-1][0]

# create directories
def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)
mkdir('logs')
mkdir(os.path.join('logs', 'val'))
mkdir(os.path.join('logs', 'debug'))
mkdir('weights')
mkdir('flows')

# find checkpoint
import path
import logger
steps = 0
if args.checkpoint is not None:
	if ':' in args.checkpoint:
		prefix, steps = args.checkpoint.split(':')
	else:
		prefix = args.checkpoint
		steps = None
	log_file, run_id = path.find_log(prefix)
	if steps is None:
		checkpoint, steps = path.find_checkpoints(run_id)[-1]
	else:
		checkpoints = path.find_checkpoints(run_id)
		try:
			checkpoint, steps = next(filter(lambda t : t[1] == steps, checkpoints))
		except StopIteration:
			print('The steps not found in checkpoints', steps, checkpoints)
			sys.stdout.flush()
			raise StopIteration
	steps = int(steps)
	if args.clear_steps:
		steps = 0
	else:
		_, exp_info = path.read_log(log_file)
		exp_info = exp_info[-1]
		for k in args.__dict__:
			if k in exp_info and k in ('tag',):
				setattr(args, k, eval(exp_info[k]))
				print('{}={}, '.format(k, exp_info[k]), end='')
		print()
	sys.stdout.flush()
# generate id
if args.checkpoint is None or args.clear_steps:
	uid = (socket.gethostname() + logger.FileLog._localtime().strftime('%b%d-%H%M') + args.gpu_device)
	tag = hashlib.sha224(uid.encode()).hexdigest()[:3]
	run_id = tag + logger.FileLog._localtime().strftime('%b%d-%H%M')
	print("run_id: {}".format(run_id))

# initiate
from network import get_pipeline
pipe = get_pipeline(args.network, ctx=ctx, config=config)
lr_schedule = dataset_cfg.optimizer.learning_rate.get(None)
if lr_schedule is not None:
	pipe.lr_schedule = lr_schedule

# load parameters from given checkpoint
if args.checkpoint is not None:
	print('Load Checkpoint {}'.format(checkpoint))
	sys.stdout.flush()
	network_class = getattr(config.network, 'class').get()
	# if train the head-stack network for the first time
	if network_class == 'MaskFlownet' and args.clear_steps and dataset_cfg.dataset.value == 'chairs':
		print('load the weight for the head network only')
		pipe.load_head(checkpoint)
	else:
		print('load the weight for the network')
		pipe.load(checkpoint)
	if network_class == 'MaskFlownet':
		print('fix the weight for the head network')
		pipe.fix_head()
	sys.stdout.flush()
	if not args.valid and not args.predict and not args.clear_steps:
		pipe.trainer.step(100, ignore_stale_grad=True)
		pipe.trainer.load_states(checkpoint.replace('params', 'states'))


#%% load training/validation datasets
validation_datasets = {}
samples = 32 if args.debug else -1

if dataset_cfg.dataset.value == 'ANHIR':
	# load the ANHIR dataset
	print('loading ANHIR dataset...')
	t0 = default_timer()
	subset = dataset_cfg.subset.value
	print(subset)
	sys.stdout.flush()
	dataset, groups, groups_train, groups_val = LoadANHIR("512", subset)
	# dataset: dict, file_name: array
	# groups, groups_train, groups_val: dict, 'id': [(img1_name, img1.csv),(img2_name, img2.csv)]
	# create the train and eval data
	train_pairs = [(f1, f2) for group in groups_train.values() for f1 in group for f2 in group if f1 is not f2]
	random.shuffle(train_pairs)
	# used for evaluation
	eval_pairs = [(f1[0], f2[0], f1[1], f2[1]) for group in groups_val.values() for f1 in group for f2 in group if f1 is not f2] # (img1_name, img2_name, img1.csv, img2.csv)
	eval_ids = ["{}_{}".format(fid, i)
            for fid in groups_val.keys() for i in range(len(groups_val[fid]))]
	eval_data = [[dataset[fid] for fid in record] for record in eval_pairs] # [img1, img2, lmk1, lmk2]
	trainSize = len(train_pairs)
	validationSize = len(eval_data)
	# used for prediction
	if args.predict:
		if args.predict_fold == 'val':
			groups_pred = groups_val
			lmk_len = 6
		elif args.predict_fold == 'train':
			groups_pred = {g:[ [k, k[:-3]+'csv'] for k in v ] for g,v in groups_train.items()}
			lmk_len = 60
		predict_data = [{
			"image_0": dataset[groups_pred[i][k][0]].transpose(1, 2, 0),
			"image_1": dataset[groups_pred[i][1-k][0]].transpose(1, 2, 0),
			"fid": "{}_{}".format(i, k),
			"lmk_0": dataset[groups_pred[i][k][1]][:lmk_len],
			"lmk_1": dataset[groups_pred[i][1-k][1]][:lmk_len],
		}	for i in groups_pred for k in [0,1] if i in ["4", "04", "6", "06"]]

else:
	raise NotImplementedError

print('Using {}s'.format(default_timer() - t0))
sys.stdout.flush()

print('data read, train {} val {}'.format(trainSize, validationSize))
sys.stdout.flush()

# ======== If to do validation ========
def validate():
	if dataset_cfg.dataset.value == "ANHIR":
		return pipe.validate(eval_data, batch_size = args.batch)

# ======== If to do prediction/visualization ========
#%% create log file
log = logger.FileLog(os.path.join(repoRoot, 'logs', 'debug' if args.debug else '', 'validate_log' if args.valid else '','{}.log'.format(run_id)))
if args.predict or args.visualize or args.valid:
	if args.predict:
		import predict
		checkpoint_name = os.path.basename(checkpoint).replace('.params', '')
		metrics = predict.predict(pipe, predict_data, os.path.join(repoRoot, 'output', args.predict_fold, checkpoint_name), batch_size=args.batch, resize = infer_resize)
		# write to csv
		save_path = os.path.join(repoRoot, 'output', args.predict_fold, checkpoint_name+'.csv')
		f = open(save_path, 'w')
		f.write(','.join(metrics[0].keys())+'\n')
		for case in metrics:
			f.write(','.join([str(v) for v in case.values()])+'\n')
		f.close()

	if args.visualize:
		import predict
		predict.visualize(pipe, predict_data)

	if args.valid:
		raw, dist_mean, dist_median, per_sample_stat  = validate()
		print("Here is the validation result:")
		print("raw: ", raw, "dist_mean: ", dist_mean, "dist_median: ", dist_median)
		log.log('steps= {} raw= {} dist_mean= {} dist_median= {}'.format(steps, raw, dist_mean, dist_median))
		eval_path = '/home/hynx/regis/SFG/SFG/baseline/logs/eval/'
		save_name = args.checkpoint + '_val.csv'
		if not os.path.exists(eval_path):
			mkdir(eval_path)
		# add fid to per_sample_stat
		per_sample_stat['fid'] = eval_ids
		# write per_sample_stat dict to csv
		with open(eval_path + save_name, 'w') as f:
			f.write(",".join(per_sample_stat.keys()) + "\n")
			for row in zip(*per_sample_stat.values()):
				f.write(",".join([str(x) for x in row]) + "\n")
	sys.exit(0)


if dataset_cfg.dataset.value == "ANHIR":
	raw_shape = dataset[train_pairs[0][0]].shape[1: 3]
else:
	raw_shape = trainImg1[0].shape[:2]

orig_shape = dataset_cfg.orig_shape.get(raw_shape)

target_shape = dataset_cfg.target_shape.get(None)
if target_shape is None:
	target_shape = [shape_axis + (64 - shape_axis) % 64 for shape_axis in orig_shape]

print(raw_shape, orig_shape, target_shape)
sys.stdout.flush()

batch_size=args.batch
assert batch_size % len(ctx) == 0
batch_size_card = batch_size // len(ctx)

log.log('start={}, train={}, val={}, host={}, batch={}'.format(steps, trainSize, validationSize, socket.gethostname(), batch_size))
information = ', '.join(['{}={}'.format(k, repr(args.__dict__[k])) for k in args.__dict__])
log.log(information)

#%% implement data augmentation
import augmentation

aug_func = augmentation.Augmentation
if args.relative == "":
	aug = aug_func(angle_range=(-17, 17), zoom_range=(0.5, 1.11), translation_range=0.1, target_shape=target_shape,
											orig_shape=orig_shape, batch_size=batch_size_card
											)
elif args.relative == "N":
	aug = aug_func(angle_range=(0, 0), zoom_range=(1, 1), translation_range=0, target_shape=target_shape,
											orig_shape=orig_shape, batch_size=batch_size_card
											)
elif args.relative == "L":
	aug = aug_func(angle_range=(-17, 17), zoom_range=(0.5, 1.11), translation_range=0.1, target_shape=target_shape,
											orig_shape=orig_shape, batch_size=batch_size_card, relative_angle=0.25, relative_scale=(0.9, 1 / 0.9)
											)
elif args.relative == "M":
	aug = aug_func(angle_range=(-17, 17), zoom_range=(0.5, 1.11), translation_range=0.1, target_shape=target_shape,
											orig_shape=orig_shape, batch_size=batch_size_card, relative_angle=0.25, relative_scale=(0.96, 1 / 0.96)
											)
elif args.relative == "S":
	aug = aug_func(angle_range=(-17, 17), zoom_range=(0.5, 1.11), translation_range=0.1, target_shape=target_shape,
											orig_shape=orig_shape, batch_size=batch_size_card, relative_angle=0.16, relative_scale=(0.98, 1 / 0.98)
											)
elif args.relative == "U":
	aug = aug_func(angle_range=(-180, 180), zoom_range=(0.5, 1.11), translation_range=0.1, target_shape=target_shape,
											orig_shape=orig_shape, batch_size=batch_size_card
											)
elif args.relative == "UM":
	aug = aug_func(angle_range=(-180, 180), zoom_range=(0.5, 1.11), translation_range=0.1, target_shape=target_shape,
											orig_shape=orig_shape, batch_size=batch_size_card,
											relative_angle=0.1, relative_scale=(0.9, 1 / 0.9), relative_translation=0.05
											)
aug.hybridize()

aug_func = augmentation.ColorAugmentation
if dataset_cfg.dataset.value == 'sintel':
	color_aug = aug_func(contrast_range=(-0.4, 0.8), brightness_sigma=0.1, channel_range=(0.8, 1.4), batch_size=batch_size_card,
		shape=target_shape, noise_range=(0, 0), saturation=0.5, hue=0.5, eigen_aug = False)
elif dataset_cfg.dataset.value == 'kitti':
	color_aug = aug_func(contrast_range=(-0.2, 0.4), brightness_sigma=0.05, channel_range=(0.9, 1.2), batch_size=batch_size_card,
		shape=target_shape, noise_range=(0, 0.02), saturation=0.25, hue=0.1, gamma_range=(-0.5, 0.5), eigen_aug = False)
else:
	color_aug = aug_func(contrast_range=(-0.4, 0.8), brightness_sigma=0.1, channel_range=(0.8, 1.4), batch_size=batch_size_card,
		shape=target_shape, noise_range=(0, 0.04), saturation=0.5, hue=0.5, eigen_aug = False)
color_aug.hybridize()

def index_generator(n):
	indices = np.arange(0, n, dtype=int)
	while True:
		np.random.shuffle(indices)
		yield from indices
train_gen = index_generator(trainSize)
class MovingAverage:
	def __init__(self, ratio=0.95):
		self.sum = 0
		self.weight = 1e-8
		self.ratio = ratio

	def update(self, v):
		self.sum = self.sum * self.ratio + v
		self.weight = self.weight * self.ratio + 1

	@property
	def average(self):
		return self.sum / self.weight

class DictMovingAverage:
	def __init__(self, ratio=0.95):
		self.sum = {}
		self.weight = {}
		self.ratio = ratio

	def update(self, v):
		for key in v:
			if key not in self.sum:
				self.sum[key] = 0
				self.weight[key] = 1e-8
			self.sum[key] = self.sum[key] * self.ratio + v[key]
			self.weight[key] = self.weight[key] * self.ratio + 1

	@property
	def average(self):
		return dict([(key, self.sum[key] / self.weight[key]) for key in self.sum])

loading_time = MovingAverage()
total_time = MovingAverage()
train_avg = DictMovingAverage()

from threading import Thread
from queue import Queue

def iterate_data(iq, gen):
	while True:
		i = next(gen)
		if dataset_cfg.dataset.value == "ANHIR":
			iq.put([dataset[fid] for fid in train_pairs[i]])


def batch_samples(iq, oq, batch_size):
	while True:
		data_batch = []
		for i in range(batch_size):
			data_batch.append(iq.get())
		oq.put([np.stack(x, axis=0) for x in zip(*data_batch)])

def remove_file(iq):
	while True:
		f = iq.get()
		try:
			os.remove(f)
		except OSError as e:
			log.log('Remove failed' + e)


data_queue = Queue(maxsize=100)
batch_queue = Queue(maxsize=4)
remove_queue = Queue(maxsize=50)

def start_daemon(thread):
	thread.daemon = True
	thread.start()

start_daemon(Thread(target=iterate_data, args=(data_queue, train_gen)))
start_daemon(Thread(target=remove_file, args=(remove_queue,)))
for i in range(2):
	start_daemon(Thread(target=batch_samples, args=(data_queue, batch_queue, batch_size)))

#%% training/ evaluation

t1 = None
checkpoints = []
maxkpval = 100
while True:
	steps += 1
	if steps%100==0:
		print("training step: {}/{}".format(steps, max_steps))

	if not pipe.set_learning_rate(steps):
		sys.exit(0)
	batch = []
	t0 = default_timer()
	if t1:
		total_time.update(t0 - t1)
	t1 = t0
	batch = batch_queue.get()
	loading_time.update(default_timer() - t0)
	img1, img2 = [batch[i] for i in range(2)]
	dist_weight = args.weight
	train_log = pipe.train_batch(dist_weight, img1, img2,color_aug, aug)

	# update log
	if steps <= 3000 or steps % 20 == 0:
		train_avg.update(train_log)
		log.log('steps={}{}, total_time={:.2f}'.format(steps, ''.join([', {}= {}'.format(k, v) for k, v in train_avg.average.items()]), total_time.average))

	# do valiation
	if steps % validation_steps == 0 or steps <= 3000:
		val_result = None
		if validationSize > 0:
			raw, eva, eva_mask, kpval = validate()
			log.log('steps= {} raw= {} kp_mean= {} kp_mean_median= {} eva_kp= {}'.format(steps, raw, eva, eva_mask, kpval))

		# save parameters
		if steps % checkpoint_steps == 0:
			if kpval < maxkpval: # gl, keep the best model for test dataset.
				maxkpval = kpval
				prefix = os.path.join(repoRoot, 'weights', '{}_{}'.format(run_id, steps))
				pipe.save(prefix)
				checkpoints.append(prefix)
				# remove the older checkpoints
				while len(checkpoints) > 3:
					prefix = checkpoints[0]
					checkpoints = checkpoints[1:]
					remove_queue.put(prefix + '.params')
					remove_queue.put(prefix + '.states')
