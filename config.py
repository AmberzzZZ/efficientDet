from backbone import swish


def default_detection_configs():
    config = {}

    config['image_size'] = 512
    config['num_classes'] = 20

    # fpn
    config['min_level'] = 3
    config['max_level'] = 7
    config['fpn_num_filters'] = 64
    config['fpn_cell_repeats'] = 3

    # anchor
    config['num_scales'] = 3
    config['aspect_ratios'] = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    config['anchor_scale'] = 4.0

    # head
    config['conv_filters'] = config['fpn_num_filters']
    config['box_class_repeats'] = 3
    config['separable_conv'] = True
    config['activation_type'] = swish
    config['survival_prob'] = None

    # classification loss (focal_loss)
    config['alpha'] = 0.25
    config['gamma'] = 1.5
    # localization loss
    config['delta'] = 0.1
    config['box_loss_weight'] = 50.0
    config['iou_loss_type'] = None         # {'iou', 'ciou', 'diou', 'giou'}
    config['iou_loss_weight'] = 1.0
    # regularization l2 loss.
    config['weight_decay'] = 4e-5

    # precision: one of 'float32', 'mixed_float16', 'mixed_bfloat16'.
    config['precision'] = None  # If None, use float32.

    # optimization
    config['momentum'] = 0.9
    config['optimizer'] = 'sgd'  # can be 'adam' or 'sgd'.
    config['learning_rate'] = 0.08  # 0.008 for adam.
    config['lr_warmup_init'] = 0.008  # 0.0008 for adam.
    config['lr_warmup_epoch'] = 1.0
    config['first_lr_drop_epoch'] = 200.0
    config['second_lr_drop_epoch'] = 250.0
    config['poly_lr_power'] = 0.9
    config['clip_gradients_norm'] = 10.0
    config['num_epochs'] = 300
    config['data_format'] = 'channels_last'

    return config




