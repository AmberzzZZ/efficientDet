from config import default_detection_configs
from backbone import EfficientNet, Conv_BN, swish
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Softmax, ReLU, add, SeparableConv2D, BatchNormalization, Activation
from keras.models import Model
import tensorflow as tf
import keras.backend as K
import numpy as np


def EfficientDet(input_tensor=None, input_shape=(512,512,3)):
    if input_tensor is not None:
        inpt = input_tensor
    else:
        inpt = Input(input_shape)

    config = default_detection_configs()

    # backbone
    x = build_backbone(inpt, config)       # [0:1xC0, 5: 32xC5]

    # feature network
    x = build_feature_network(x, config)        # [P3", P7"]

    # heads
    class_outputs, box_outputs = build_class_and_box_outputs(x, config)

    # list(class_outputs.values())+list(box_outputs.values())
    model = Model(inpt, list(class_outputs.values())+list(box_outputs.values()))

    return model


def build_backbone(x, config):
    _, features = EfficientNet(x)
    # level3-level7 features (8x-128x)
    return {0: x, 1:features[0], 2:features[1], 3:features[2], 4:features[4], 5:features[6]}


def build_feature_network(x, config):
    feats = []
    for level in range(config['min_level'], config['max_level']+1):   # [C3,C7]
        if level in x.keys():
            feats.append(x[level])
        else:
            # Adds a coarser level by downsampling the last feature map
            target_h = (feats[-1]._keras_shape[1] - 1)//2 + 1
            target_w = (feats[-1]._keras_shape[2] - 1)//2 + 1
            target_c = config['fpn_num_filters']
            feats.append(resample_feature_map(feats[-1], target_h, target_w, target_c))

    for i in range(config['fpn_cell_repeats']):
        feats = build_bifpn(feats, config)
    return feats


def build_class_and_box_outputs(x, config):
    num_anchors = len(config['aspect_ratios']) * config['num_scales']
    class_outputs = {}
    box_outputs = {}
    for level in range(config['min_level'], config['max_level']+1):
        class_outputs[level] = class_net(x[level-config['min_level']],
                                         n_classes=config['num_classes'],
                                         n_anchors=num_anchors,
                                         n_filters=config['fpn_num_filters'],
                                         act_type=config['activation_type'],
                                         repeats=config['box_class_repeats'],
                                         survival_prob=config['survival_prob'])
    #     box_outputs[level] = box_net(x[level-config['min_level']],
    #                                  n_anchors=num_anchors,
    #                                  n_filters=config['fpn_num_filters'],
    #                                  act_type=config['activation_type'],
    #                                  repeats=config['box_class_repeats'],
    #                                  survival_prob=config['survival_prob'])
    return class_outputs, box_outputs


def class_net(x, n_classes, n_anchors, n_filters, act_type,
              repeats=4, separable_conv=True, survival_prob=None):
    # conv-bn-swish + id
    for i in range(repeats):
        inpt = x
        if separable_conv:
            x = SeparableConv2D(n_filters, kernel_size=3, strides=1, padding='same',
                                depthwise_initializer=tf.initializers.variance_scaling(),
                                pointwise_initializer=tf.initializers.variance_scaling(),
                                bias_initializer=tf.zeros_initializer())(x)
        else:
            x = Conv2D(n_filters, kernel_size=3, strides=1, padding='same',
                       kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                       bias_initializer=tf.zeros_initializer())(x)
        x = BatchNormalization()(x)
        x = Activation(act_type)(x)
        if i>0 and survival_prob:
            x = Lambda(drop_connect, arguments={'survival_prob': survival_prob, 'is_training': True})(x)
        x = add([x, inpt])

    # head
    x = Conv2D(n_classes*n_anchors, kernel_size=3, strides=1, padding='same',
               bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)))(x)

    return x


def box_net(x, n_anchors, n_filters, act_type,
            repeats=4, separable_conv=True, survival_prob=None):
    # conv-bn-swish + id
    for i in range(repeats):
        inpt = x
        if separable_conv:
            x = SeparableConv2D(n_filters, kernel_size=3, strides=1, padding='same',
                                depthwise_initializer=tf.initializers.variance_scaling(),
                                pointwise_initializer=tf.initializers.variance_scaling(),
                                bias_initializer=tf.zeros_initializer())(x)
        else:
            x = Conv2D(n_filters, kernel_size=3, strides=1, padding='same',
                       kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                       bias_initializer=tf.zeros_initializer())(x)
        x = BatchNormalization()(x)
        x = Activation(act_type)(x)
        if i>0 and survival_prob:
            x = Lambda(drop_connect, arguments={'survival_prob': survival_prob, 'is_training': True})(x)
        x = add([x, inpt])

    # head
    x = Conv2D(4*n_anchors, kernel_size=3, strides=1, padding='same',
               bias_initializer=tf.zeros_initializer())(x)

    return x


def drop_connect(x, survival_prob, is_training=False):
    # Unlike conventional way that multiply survival_prob at test time, here we
    # divide survival_prob at training time, such that no addition compute is
    # needed at test time.
    if not is_training:
        return x

    # Compute tensor.
    batch_size = tf.shape(x)[0]
    random_tensor = survival_prob
    random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=x.dtype)
    binary_tensor = tf.floor(random_tensor)
    x = tf.div(x, survival_prob) * binary_tensor
    return x


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
        new_node = Lambda(fuse_features, arguments={'weight_method': 'fast'})(nodes_in)
        new_node = Conv_BN(new_node, config['fpn_num_filters'], activation=None)
        feats.append(new_node)

    fpn_feats = feats[-5:]
    fpn_feats.reverse()

    return fpn_feats         # [P3", P7"]


def fuse_features(nodes_in, weight_method='fast'):
    if weight_method=='softmax':
        weights = [tf.Variable(1.) for i in nodes_in]
        normed_weights = tf.unstack(K.softmax(tf.stack(weights)))
        new_node = add([nodes_in[i] * normed_weights[i] for i in range(len(nodes_in))])
    elif weight_method=='fast':
        weights = [ReLU()(tf.Variable(1.)) for i in nodes_in]
        normed_weights = tf.unstack((tf.stack(weights)/(K.sum(weights)+K.epsilon())))
        new_node = add([nodes_in[i] * normed_weights[i] for i in range(len(nodes_in))])
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
        x = Lambda(tf.image.resize_nearest_neighbor, arguments={'size': [target_h, target_w]})(x)
    else:
        print("Incompatible target feature map size")
    return x


if __name__ == '__main__':

    model = EfficientDet(input_tensor=None)
    model.summary()






