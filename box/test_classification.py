import cv2
import numpy as np
from keras.layers import concatenate
from keras.models import Model
import keras.backend as K
from efficientDet import EfficientDet
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))


def sigmoid(x):
    return 1 / (1 + np.exp(1e-9-x))


if __name__ == '__main__':

    test_file = "./data/tux_hacking.png"
    num_classes = 15
    scores_thresh = 0.2
    strides = [8, 16, 32]

    # base model
    model = EfficientDet(input_shape=(512,512,3))
    model.load_weights("effd4.h5")

    # add postprocess tail: sigmoid
    layer_idx = 33
    cls_outputs = ['reshape_%s' % str(i*2+layer_idx) for i in range(len(strides))]
    cls_outputs = [model.get_layer(name=n).output for n in cls_outputs]
    box_outputs = ['reshape_%s' % str(i*2+1+layer_idx) for i in range(len(strides))]
    box_outputs = [model.get_layer(name=n).output for n in box_outputs]
    outputs = [concatenate([i,j]) for i,j in zip(box_outputs, cls_outputs)]
    model = Model(model.input[0], outputs)

    # test
    img = cv2.imread(test_file, 1)
    if np.max(img)>1:
        img = img / 255.
    inpt = cv2.resize(img, (512,512))
    inpt = np.reshape(inpt, (1,512,512,3))

    outputs = model.predict(inpt)

    probs = [0. for i in range(num_classes)]
    for i in range(len(strides)):
        h = 512 / strides[i]
        for id in range(num_classes):
            max_prob = np.max(sigmoid(outputs[i][0,:,:,:,4+id]))
            cls_idx = np.argmax(sigmoid(outputs[i][0,:,:,:,4+id]))
            grid_y = cls_idx // (h*n_anchors)
            grid_x = (cls_idx - grid_y*h*n_anchors) // (n_anchors)
            a = cls_idx - grid_y*h*n_anchors - grid_x*n_anchors
            grid_y, grid_x, a = int(grid_y), int(grid_x), int(a)
            offset = outputs[i][0,grid_y,grid_x,a]
            anchor = anchors[i][a]
            if max_prob>probs[id]:
                probs[id] = max_prob
                xc = offset[0]*anchor[0] + (grid_x+0.5)*strides[i]
                yc = offset[1]*anchor[1] + (grid_y+0.5)*strides[i]
                pred_w = np.exp(offset[2])*anchor[0]
                pred_h = np.exp(offset[3])*anchor[1]
    topN = np.argsort(probs)[::-1]
    print(probs)
    print(topN)
    print("prediction: ", topN[0], max(probs))
























