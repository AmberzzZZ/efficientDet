from backbone import swish


def default_detection_configs():
    config = {}

    # model
    config['input_shape'] = (512,512,1)
    config['width_coefficient'] = 1.4
    config['depth_coefficient'] = 1.8
    config['dropout_rate'] = 0.4
    config['pretrained'] = "weights/effb4_ch1.h5"

    config['num_classes'] = 20
    config['num_points'] = 2

    # fpn
    config['min_level'] = 3
    config['max_level'] = 7
    config['fpn_num_filters'] = 64
    config['fpn_cell_repeats'] = 1

    # head
    config['conv_filters'] = config['fpn_num_filters']
    config['min_out_level'] = 3
    config['max_out_level'] = 5
    config['box_class_repeats'] = 3
    config['separable_conv'] = True
    config['activation_type'] = swish
    config['survival_prob'] = None

    return config










