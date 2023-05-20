#%% import packages
import os
import sys
from timeit import default_timer

import configure
import func_utils
import mxnet as mx
import numpy as np

#%% parse arguments
configure.set_gpu()
configure.set_env()
args = configure.get_parse_args()
cfg = configure.get_config(args)
cfg.ctx = [mx.cpu()] if cfg.gpu_device == '' else [mx.gpu(gpu_id) for gpu_id in map(int, cfg.gpu_device.split(','))]
cfg.infer_resize = [int(s) for s in cfg.resize.split(',')] if cfg.resize else None
cfg.repoRoot = r'.'

#%% get configs and set up dir, log
## config of networks, data loaders, and logging
network_cfg, dataset_cfg = configure.read_config_file(cfg)
cfg.validation_steps = dataset_cfg.validation_steps.value
cfg.checkpoint_steps = dataset_cfg.checkpoint_steps.value
cfg.max_steps = network_cfg.optimizer.learning_rate.value[-1][0]

# set up names and files
configure.setup_saving_dir(cfg)
cfg.checkpoint_stem = cfg.checkpoint
cfg.checkpoint, cfg.steps = configure.find_checkpoint(cfg)
run_id = configure.generate_runid(cfg); cfg.run_id = run_id
log = configure.setup_log(cfg)

#%% build pipeline and dataset
# initiate pipe
pipe = configure.get_pipe(cfg, network_cfg, dataset_cfg)
configure.load_pipe(cfg, pipe, network_cfg, dataset_cfg)
if cfg.vp:
    configure.set_vp(cfg, pipe)

# load training/validation datasets
dataset, train_pairs, eval_data, eval_ids, predict_data = configure.load_dataset(cfg, dataset_cfg)

#%%  If to do validation/prediction/visualization
def validate():
    if dataset_cfg.dataset.value == "ANHIR":
        return pipe.validate(eval_data, batch_size = cfg.batch)

if cfg.predict or cfg.visualize or cfg.valid or cfg.inversion_test:
    if cfg.inversion_test:
        import inv_test as T
        img1, img2 = eval_data[0][:2]
        # give it batch axis
        img1, img2 = [img1.transpose(1,2,0)], [img2.transpose(1,2,0)]

        flow, _, warped = next(pipe.predict(img1, img2, 1, ret_orig_warp=True))
        warped = [warped]
        rev_flow, _, rev_img2 = next(pipe.predict(img2, warped, 1, ret_orig_warp=True))
        test = T.Test()
        flow, rev_flow = flow[None].transpose(0,3,1,2), rev_flow[None].transpose(0,3,1,2)
        vis_compose_flow = test.test(flow, rev_flow)
        comp_flow = test.composite_flow(flow, rev_flow)

        mx_img2 = mx.nd.array(img2[0].transpose(2,0,1)).as_in_context(mx.gpu())[None]
        comp_warped = pipe.reconstruction(mx_img2, comp_flow).asnumpy()

        # visualize the img of composite flow
        img2, rev_img2 = img2[0].transpose(2,0,1), rev_img2.transpose(2,0,1)
        import vis_utils as V
        img_dict = {
            'img2': img2,
            'img1': img1[0].transpose(2,0,1),
            'warp': warped[0].transpose(2,0,1),
            'rev_img2': rev_img2,
            'comp_flow': vis_compose_flow,
            'comp_warped': comp_warped[0]
        }
        for k in img_dict:
            img_k = V.arr2img(img_dict[k])
            img_k.save('tmp/{}.png'.format(k))
        breakpoint()

    if cfg.predict:
        import predict
        checkpoint_name = os.path.basename(cfg.checkpoint).replace('.params', '')
        bd = os.path.join(cfg.repoRoot, 'output/pred', cfg.predict_fold,)
        metrics = predict.predict(pipe, predict_data,  os.path.join(bd, checkpoint_name), batch_size=cfg.batch, resize = cfg.infer_resize)
        # write to csv
        save_path = os.path.join(bd, checkpoint_name+'.csv')
        # mkdir if not exist
        configure.mkdir(os.path.dirname(save_path))
        f = open(save_path, 'w')
        f.write(','.join(metrics[0].keys())+'\n')
        for case in metrics:
            f.write(','.join([str(v) for v in case.values()])+'\n')
        f.close()

    if cfg.visualize:
        import predict
        predict.visualize(pipe, predict_data)

    if cfg.valid:
        raw, dist_mean, dist_median, per_sample_stat  = validate()
        print("Here is the validation result:")
        print("raw: ", raw, "dist_mean: ", dist_mean, "dist_median: ", dist_median)
        log.log('steps= {} raw= {} dist_mean= {} dist_median= {}'.format(cfg.steps, raw, dist_mean, dist_median))
        eval_path = '/home/hynx/regis/SFG/SFG/baseline/output/valid/'
        save_name = cfg.checkpoint_stem + '_val.csv'
        if not os.path.exists(eval_path): os.makedirs(eval_path)
        # add fid to per_sample_stat
        per_sample_stat['fid'] = eval_ids
        # write per_sample_stat dict to csv
        with open(eval_path + save_name, 'w') as f:
            f.write(",".join(per_sample_stat.keys()) + "\n")
            for row in zip(*per_sample_stat.values()):
                f.write(",".join([str(x) for x in row]) + "\n")
    sys.exit(0)

#%% set training parameters
# get dataset shape
raw_shape, orig_shape, target_shape, batch_size, batch_size_card \
    = func_utils.get_shape(cfg, dataset_cfg, dataset, train_pairs, log)

# get augmenters
aug, color_aug = func_utils.get_augmentation(cfg, dataset_cfg)
# get time logger
loading_time, total_time, train_avg = configure.get_time_logger()

# set dataloader
batch_queue, remove_queue = configure.get_batch_queue(cfg, dataset_cfg, dataset, train_pairs, log)

#%% training
t1 = None
checkpoints = []
maxkpval = 100
steps = cfg.steps
while True:
    steps += 1
    if steps%100==0:
        print("training step: {}/{}".format(steps, cfg.max_steps))

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
    dist_weight = cfg.weight
    train_log = pipe.train_batch(img1, img2,color_aug, aug)

    # update log
    if steps <= 3000 or steps % 20 == 0:
        train_avg.update(train_log)
        log.log('steps={}{}, total_time={:.2f}'.format(steps, ''.join([', {}= {}'.format(k, v) for k, v in train_avg.average.items()]), total_time.average))

    # do valiation
    if steps % cfg.validation_steps == 0 or steps <= 3000:
        val_result = None
        if cfg.validationSize > 0:
            raw, eva, eva_mask, _ = validate()
            kpval = eva
            log.log('steps= {} raw= {} kp_mean= {} kp_mean_median= {}'.format(steps, raw, eva, eva_mask))

        # save parameters
        if steps % cfg.checkpoint_steps == 0:
            if kpval < maxkpval: # gl, keep the best model for test dataset.
                maxkpval = kpval
                prefix = os.path.join(cfg.repoRoot, 'weights', run_id, '{}_{}'.format(run_id, steps))
                pipe.save(prefix)
                checkpoints.append(prefix)
                # remove the older checkpoints
                while len(checkpoints) > 3:
                    prefix = checkpoints[0]
                    checkpoints = checkpoints[1:]
                    remove_queue.put(prefix + '.params')
                    remove_queue.put(prefix + '.states')
