from backbone import swish
import numpy as np


def default_detection_configs():
    config = {}

    # model architecture
    config['image_size'] = 512
    config['width_coefficient'] = 1.4
    config['depth_coefficient'] = 1.8
    config['dropout_rate'] = 0.2
    config['min_level'] = 3
    config['max_level'] = 7
    config['strides'] = [8,16,32]
    config['num_classes'] = 15
    # anchor
    config['aspect_ratios'] = [1.0, 0.5, 2.0]
    config['anchor_scale'] = [2**0, 2**(1/3.), 2**(2/3.)]
    config['size'] = {8:32, 16:64, 32:128, 64:256, 128:512}
    config['anchors'] = get_anchors(config['aspect_ratios'], config['anchor_scale'], config['size'],
                            strides=[2**i for i in range(config['min_level'], config['max_level']+1)])
    config['n_anchors'] = 9
    # config['anchors'] = "anchors.txt"

    # fpn
    config['fpn_num_filters'] = 128
    config['fpn_cell_repeats'] = 5
    # head
    config['box_class_repeats'] = 4
    config['conv_filters'] = config['fpn_num_filters']
    config['separable_conv'] = True
    config['activation_type'] = swish
    config['survival_prob'] = None


    return config


def get_anchors(anchor_ratios, anchor_scales, anchor_sizes, strides):
    anchors = []
    for s in strides:
        base_size = anchor_sizes[s]
        anchors_s = base_size * np.tile(np.expand_dims(anchor_scales,axis=-1), (len(anchor_ratios),2)).astype(np.float32)
        anchors_s[:,0] = anchors_s[:,0] / np.sqrt(np.repeat(anchor_ratios, len(anchor_scales)))
        anchors_s[:,1] = anchors_s[:,0] * np.repeat(anchor_ratios, len(anchor_scales))
        anchors.append(anchors_s)

    return anchors


