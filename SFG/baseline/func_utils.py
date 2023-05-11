#%% implement data augmentation
import socket
import sys
import augmentation
def get_augmentation(cfg, dataset_cfg):
    target_shape, orig_shape, raw_shape = cfg.target_shape, cfg.orig_shape, cfg.raw_shape
    batch_size, batch_size_card = cfg.batch_size, cfg.batch_size_card
    aug_func = augmentation.Augmentation
    if cfg.relative == "":
        aug = aug_func(angle_range=(-17, 17), zoom_range=(0.5, 1.11), translation_range=0.1, target_shape=target_shape,
                                                orig_shape=orig_shape, batch_size=batch_size_card
                                                )
    elif cfg.relative == "N":
        aug = aug_func(angle_range=(0, 0), zoom_range=(1, 1), translation_range=0, target_shape=target_shape,
                                                orig_shape=orig_shape, batch_size=batch_size_card
                                                )
    elif cfg.relative == "L":
        aug = aug_func(angle_range=(-17, 17), zoom_range=(0.5, 1.11), translation_range=0.1, target_shape=target_shape,
                                                orig_shape=orig_shape, batch_size=batch_size_card, relative_angle=0.25, relative_scale=(0.9, 1 / 0.9)
                                                )
    elif cfg.relative == "M":
        aug = aug_func(angle_range=(-17, 17), zoom_range=(0.5, 1.11), translation_range=0.1, target_shape=target_shape,
                                                orig_shape=orig_shape, batch_size=batch_size_card, relative_angle=0.25, relative_scale=(0.96, 1 / 0.96)
                                                )
    elif cfg.relative == "S":
        aug = aug_func(angle_range=(-17, 17), zoom_range=(0.5, 1.11), translation_range=0.1, target_shape=target_shape,
                                                orig_shape=orig_shape, batch_size=batch_size_card, relative_angle=0.16, relative_scale=(0.98, 1 / 0.98)
                                                )
    elif cfg.relative == "U":
        aug = aug_func(angle_range=(-180, 180), zoom_range=(0.5, 1.11), translation_range=0.1, target_shape=target_shape,
                                                orig_shape=orig_shape, batch_size=batch_size_card
                                                )
    elif cfg.relative == "UM":
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
    return aug, color_aug

#%%
def get_shape(cfg, dataset_cfg, dataset, train_pairs, log):
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
    batch_size=cfg.batch
    assert batch_size % len(cfg.ctx) == 0
    batch_size_card = batch_size // len(cfg.ctx)
    cfg.raw_shape, cfg.orig_shape, cfg.target_shape = raw_shape, orig_shape, target_shape
    cfg.batch_size, cfg.batch_size_card = batch_size, batch_size_card

    log.log('start={}, train={}, val={}, host={}, batch={}'.format(cfg.steps, cfg.trainSize, cfg.validationSize, socket.gethostname(), batch_size))
    information = ', '.join(['{}={}'.format(k, repr(cfg.__dict__[k])) for k in cfg.__dict__])
    log.log(information)

    return raw_shape, orig_shape, target_shape, batch_size, batch_size_card