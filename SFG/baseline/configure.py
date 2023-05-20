import argparse
import hashlib
import os
import random
import socket
import sys
from timeit import default_timer

import logger
import ml_collections
import network.config
import yaml
from network import get_pipeline
from reader.ANHIR import ANHIRPredict, LoadANHIR

#%% get args and configs
# setup GPU
def set_gpu():
    import subprocess
    GPU_ID = subprocess.getoutput('nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1| head -n 1 | xargs')
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

def set_env():
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

# get parse args
def get_parse_args():
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

    # parser for volume-preserving
    parser.add_argument('--vp', action='store_true', help='Do volume-preserving')
    parser.add_argument('-it', '--inversion_test', action='store_true', help='Do inversion test')

    args = parser.parse_args()
    return args

# get config
def get_config(args):
    # convert args to ml collection
    config = ml_collections.ConfigDict(vars(args))
    return config

# read config file
def read_config_file(cfg):

    repoRoot = cfg.repoRoot
    # load network configuration
    with open(os.path.join(repoRoot, 'network', 'config', cfg.config)) as f:
        network_config =  network.config.Reader(yaml.safe_load(f))
    # load training configuration
    with open(os.path.join(repoRoot, 'network', 'config', cfg.dataset_cfg)) as f:
        dataset_cfg = network.config.Reader(yaml.safe_load(f))
    return network_config, dataset_cfg

#%% set up dir and files
def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)
def setup_saving_dir(cfg):
   # create directories
    mkdir('logs')
    mkdir(os.path.join('logs', 'debug'))
    mkdir('weights')
    mkdir('flows')

def setup_log(cfg):
    mode = 'validate_log' if cfg.valid else ''
    log_path = os.path.join(cfg.repoRoot, 'logs', 'debug' if cfg.debug else '', '{}.log'.format(cfg.run_id))
    print('Log file: {}'.format(log_path))
    return logger.FileLog(log_path)

#%% pipe setup
def get_pipe(cfg, network_cfg, dataset_cfg):
    pipe = get_pipeline(cfg.network, ctx=cfg.ctx, config=network_cfg)
    lr_schedule = dataset_cfg.optimizer.learning_rate.get(None)
    if lr_schedule is not None:
        pipe.lr_schedule = lr_schedule
    return pipe

def find_checkpoint(cfg):
    import path
    steps = 0
    if cfg.checkpoint is not None:
        if ':' in cfg.checkpoint:
            prefix, steps = cfg.checkpoint.split(':')
        else:
            prefix = cfg.checkpoint
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
        if cfg.clear_steps:
            steps = 0
        else:
            _, exp_info = path.read_log(log_file)
            exp_info = exp_info[-1]
            for k in cfg.__dict__:
                if k in exp_info and k in ('tag',):
                    setattr(cfg, k, eval(exp_info[k]))
                    print('{}={}, '.format(k, exp_info[k]), end='')
            print()
        sys.stdout.flush()
        return checkpoint, steps
    else:
        return None, 0

def generate_runid(cfg):
    if cfg.checkpoint is None or cfg.clear_steps:
        uid = (socket.gethostname() + logger.FileLog._localtime().strftime('%b%d-%H%M') + cfg.gpu_device)
        tag = hashlib.sha224(uid.encode()).hexdigest()[:3]
        run_id = tag + logger.FileLog._localtime().strftime('%b%d-%H%M')
        print("run_id: {}".format(run_id))
        return run_id
    return None

def load_pipe(cfg, pipe, network_cfg, dataset_cfg):
    # load parameters from given checkpoint
    if cfg.checkpoint is not None:
        print('Load Checkpoint {}'.format(cfg.checkpoint))
        sys.stdout.flush()
        network_class = getattr(network_cfg.network, 'class').get()
        # if train the head-stack network for the first time
        if network_class == 'MaskFlownet' and cfg.clear_steps and dataset_cfg.dataset.value == 'chairs':
            print('load the weight for the head network only')
            pipe.load_head(cfg.checkpoint)
        else:
            print('load the weight for the network')
            pipe.load(cfg.checkpoint)
        if network_class == 'MaskFlownet':
            print('fix the weight for the head network')
            pipe.fix_head()
        sys.stdout.flush()
        if not cfg.valid and not cfg.predict and not cfg.clear_steps:
            pipe.trainer.step(100, ignore_stale_grad=True)
            pipe.trainer.load_states(cfg.checkpoint.replace('params', 'states'))

#%% load dataset
def load_dataset(cfg, dataset_cfg):
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
        cfg.trainSize = len(train_pairs)
        cfg.validationSize = len(eval_data)
        # used for prediction
        if cfg.predict:
            if cfg.predict_fold == 'val':
                groups_pred = groups_val
                lmk_len = 6
            elif cfg.predict_fold == 'train':
                groups_pred = {g:[ [k, k[:-3]+'csv'] for k in v ] for g,v in groups_train.items()}
                lmk_len = 60
            predict_data = [{
                "image_0": dataset[groups_pred[i][k][0]].transpose(1, 2, 0),
                "image_1": dataset[groups_pred[i][1-k][0]].transpose(1, 2, 0),
                "fid": "{}_{}".format(i, k),
                "lmk_0": dataset[groups_pred[i][k][1]][:lmk_len],
                "lmk_1": dataset[groups_pred[i][1-k][1]][:lmk_len],
            }	for i in groups_pred for k in [0,1] ]#if i in ["4", "04", "6", "06"]]
        else: predict_data = None

    else:
        raise NotImplementedError

    print('Using {}s'.format(default_timer() - t0))
    sys.stdout.flush()

    print('data read, train {} val {}'.format(cfg.trainSize, cfg.validationSize))
    sys.stdout.flush()
    return dataset, train_pairs, eval_data, eval_ids, predict_data

#%% time metric
def get_time_logger():
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
    return loading_time, total_time, train_avg

#%% set data loader
def get_batch_queue(cfg, dataset_cfg, dataset, train_pairs, log):
    batch_size = cfg.batch_size
    import numpy as np
    def index_generator(n):
        indices = np.arange(0, n, dtype=int)
        while True:
            np.random.shuffle(indices)
            yield from indices
    train_gen = index_generator(cfg.trainSize)


    from queue import Queue
    from threading import Thread


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
    return batch_queue, remove_queue