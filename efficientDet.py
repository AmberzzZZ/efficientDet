from config import default_detection_configs
from backbone import EfficientNet, Conv_BN
from keras.layers import Conv2D, MaxPooling2D, Lambda, Softmax, ReLU, add
import tensorflow as tf
import keras.backend as K


def EfficientDet(input_tensor):

    config = default_detection_configs()

    # backbone
    x = build_backbone(input_tensor, config)

    # feature network
    x = build_feature_network(x, config)

    # head
    class_outputs, box_outputs = build_class_and_box_outputs(x, config)

    return class_outputs, box_outputs



def build_backbone(x, config):
    _, features = EfficientNet(x)
    # level3-level7 features (8x-128x)
    return {0: x, 1:features[0], 2:features[1], 3:features[2], 4:features[4], 5:features[6]}


def build_feature_network(x, config):
    feats = []
    for level in range(config['min_level'], config['max_level']+1):   # [3,7]
        if level in x.keys():
            feats.append(level)
        else:
            # Adds a coarser level by downsampling the last feature map
            target_h = (feats[-1]._keras_shape[1] - 1)//2 + 1
            target_w = (feats[-1]._keras_shape[2] - 1)//2 + 1
            target_c = config['fpn_num_filters']
            feats.append(resample_feature_map(feats[-1], target_h, target_w, target_c))

    for i in range(config['fpn_cell_repeats']):
        feats = build_bifpn(feats, config)
    return feats


def build_class_and_box_outputs():
    pass


def build_bifpn(feats, config):
    # Node id starts from the input features and monotonically increase whenever
    # a new node is added. Here is an example for level P3 - P7:
    #     P7 (4)              P7" (12)
    #     P6 (3)    P6' (5)   P6" (11)
    #     P5 (2)    P5' (6)   P5" (10)
    #     P4 (1)    P4' (7)   P4" (9)
    #     P3 (0)              P3" (8)
    fpn_nodes = [{'feat_level': 6, 'inputs_offsets': [3, 4]},  # for P6'
                 {'feat_level': 5, 'inputs_offsets': [2, 5]},  # for P5'
                 {'feat_level': 4, 'inputs_offsets': [1, 6]},  # for P4'
                 {'feat_level': 3, 'inputs_offsets': [0, 7]},  # for P3"
                 {'feat_level': 4, 'inputs_offsets': [1, 7, 8]},  # for P4"
                 {'feat_level': 5, 'inputs_offsets': [2, 6, 9]},  # for P5"
                 {'feat_level': 6, 'inputs_offsets': [3, 5, 10]},  # for P6"
                 {'feat_level': 7, 'inputs_offsets': [4, 11]}]  # for P7"
    for i, fpn_node in enumerate(fpn_nodes):
        new_node_h, new_node_w = feats[fpn_node['feat_level']-config['min_level']]._keras_shape[1:3]
        nodes_in = []
        for idx, inputs_offset in enumerate(fpn_node['inputs_offsets']):
            input_node = feats[inputs_offset]
            input_node = resample_feature_map(input_node, new_node_h, new_node_w, config['fpn_num_filters'])
            nodes_in.append(input_node)
        new_node = fuse_features(nodes_in, weight_mode='fast')
        new_node = Conv_BN(new_node, config['fpn_num_filters'], activation=None)
        feats.append(new_node)

    fpn_feats = []
    for i in range(config['min_level'], config['max_level']+1):
        for j, fpn_node in enumerate(reversed(fpn_nodes)):
            if fpn_node['feat_level'] == i:
                fpn_feats = feats[-1-i]
                break
    return fpn_feats         # [P3", P7"]


def fuse_features(nodes_in, weight_method='fast'):
    if weight_method=='softmax':
        weights = [tf.Variable(1.) for i in nodes_in]
        normed_weights = Softmax()(weights)
        new_node = K.sum(nodes_in * normed_weights)
    elif weight_method=='fast':
        weights = [tf.Variable(1.) for i in nodes_in]
        normed_weights = ReLU()(weights)
        normed_weights = normed_weights / (K.sum(normed_weights) + K.epsilon())
        new_node = K.sum(nodes_in * normed_weights)
    else:  # normal sum
        new_node = add(nodes_in)
    return new_node


def resample_feature_map(x, target_h, target_w, target_c):
    # 1x1 conv if channel not match, conv-bn-swish
    x = Conv_BN(x, target_c, kernel_size=1, strides=1)
    # resize
    h, w = x._keras_shape[1:3]
    if h > target_h and w > target_w:
        h_stride = int((h - 1)//target_h + 1)
        w_stride = int((w - 1)//target_w + 1)
        x = MaxPooling2D(pool_size=(h_stride+1, w_stride+1), strides=(h_stride,w_stride), padding='same')(x)
    elif h <= target_h and w <= target_w:
        x = Lambda(tf.image.resize_nearest_neighbor, args={'size': [target_h, target_w]})(x)
    else:
        print("Incompatible target feature map size")
    return x








