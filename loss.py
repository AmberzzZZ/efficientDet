# total_loss: cls_loss, box_loss, box_iou_loss
import tensorflow as tf
import keras.backend as K
import math


def det_loss(args, config):
    n_levels = len(args)//4
    cls_preds = args[:n_levels]
    cls_true = args[2*n_levels:3*n_levels]
    box_preds = args[n_levels:2*n_levels]
    box_true = args[3*n_levels:]
    cls_losses = []
    box_reg_losses = []
    box_iou_losses = []
    for i in range(n_levels):
        cls_loss_i = cls_loss(cls_preds[i], cls_true[i], config['alpha'], config['gamma'])
        cls_losses.append(K.sum(cls_loss_i))
        box_reg_loss_i = box_reg_loss(box_preds[i], box_true[i], config['delta'])
        box_reg_losses.append(K.sum(box_reg_loss_i))
        box_iou_loss_i = box_iou_loss(box_preds[i], box_true[i], config['iou_loss_type'])
        box_iou_losses.append(K.sum(box_iou_loss_i))

    cls_losss = tf.add_n(cls_losses)
    box_reg_losss = tf.add_n(box_reg_losses)
    box_iou_losss = tf.add_n(box_iou_losses) if box_iou_losses else 0.0

    total_loss = cls_losss + config['box_loss_weight']*box_reg_losss + config['iou_loss_weight']*box_iou_losss

    # return total_loss, cls_loss, box_reg_loss, box_iou_loss
    return total_loss


# for each resolution
def cls_loss(cls_pred, cls_true, alpha=0.25, gamma=2.0):
    # focal loss (1-p_t)^r * log(pt)
    pos_mask = tf.equal(cls_true, 1.0)
    # first half: stable form, exp(-r*z*x - r*log(1+exp(-x)))
    first_half = K.exp(gamma*cls_true*(-cls_pred) - gamma*K.log(1+K.exp(-cls_pred)))
    # second half: log(pt), sigmoid bce
    second_half = tf.where(pos_mask, K.log(K.sigmoid(cls_pred)), K.log(1-K.sigmoid(cls_pred)))
    loss = first_half * second_half
    focal_loss = tf.where(pos_mask, alpha*loss, (1-alpha)*loss)
    return focal_loss


# for each resolution
def box_reg_loss(box_pred, box_true, delta=0.1):
    # huber loss
    valid_mask = tf.not_equal(box_true, 0.0)
    huber_loss = tf.where((K.abs(box_true-box_pred)<delta) & valid_mask,
                          0.5*((box_true-box_pred)**2),
                          delta*K.abs(box_true-box_pred) - 0.5*(delta**2))
    return huber_loss


# for each resolution
def box_iou_loss(box_pred, box_true, iou_type):
    box_pred_list = tf.unstack(box_pred, None, axis=-1)
    box_true_list = tf.unstack(box_true, None, axis=-1)
    iou_list = []
    for i in range(0, len(box_pred_list), 4):
        box_pred_i = box_pred_list[i:i+4]          # [N,H,W,1] * 4
        box_true_i = box_true_list[i:i+4]
        g_ymin, g_xmin, g_ymax, g_xmax = box_true_i
        valid_mask = tf.cast(tf.not_equal((g_ymax-g_ymin)*(g_xmax-g_xmin), 0), tf.float32)       # [N,H,W,1]
        box_pred_i = [i*valid_mask for i in box_pred_i]
        iou_list.append(cal_iou(box_pred_i, box_true_i, iou_type))
    return K.sum(1. - tf.stack(iou_list))


# for each featuremap-boxes
def cal_iou(box_pred, box_true, iou_type):
    # IoU = |A&B| / |A|B|
    # GIoU = IoU - |C - A|B|/C
    # DIoU = IoU - l2(Center_A, Center_B) / diagonal_C^2
    # CIoU = IoU - DIoU - a * v, a is a positive trade-off parameter, v = (arctan(w_gt / h_gt) - arctan(w / h)) * 4 / pi^2
    g_ymin, g_xmin, g_ymax, g_xmax = box_true
    p_ymin, p_xmin, p_ymax, p_xmax = box_pred
    p_width, p_height = K.maximum(0., p_xmax - p_xmin), K.maximum(0., p_ymax - p_ymin)
    p_area = p_width * p_height
    g_width, g_height = K.maximum(0., g_xmax - g_xmin), K.maximum(0., g_ymax - g_ymin)
    g_area = g_width * g_height
    intersect_ymin, intersect_xmin = K.maximum(p_ymin, g_ymin), K.maximum(p_xmin, g_xmin)
    intersect_ymax, intersect_xmax = K.minimum(p_ymax, g_ymax), K.minimum(p_xmax, g_xmax)
    intersect_width, intersect_height = K.maximum(0., intersect_xmax - intersect_xmin), K.maximum(0., intersect_ymax - intersect_ymin)
    intersect_area = intersect_width * intersect_height
    union_area = p_area + g_area - intersect_area
    iou = intersect_area / union_area
    if iou_type == 'iou':
        return iou
    C_ymin, C_xmin = K.minimum(p_ymin, g_ymin), K.minimum(p_xmin, g_xmin)
    C_ymax, C_xmax = K.maximum(p_ymax, g_ymax), K.maximum(p_xmax, g_xmax)
    C_width, C_height = K.maximum(0., C_xmax - C_xmin), K.maximum(0., C_ymax - C_ymin)
    C_area = C_width * C_height
    giou = iou - (C_area-union_area)/C_area
    if iou_type == 'giou':
        return giou
    p_center = tf.stack([(p_ymin + p_ymax) / 2, (p_xmin + p_xmax) / 2])
    g_center = tf.stack([(g_ymin + g_ymax) / 2, (g_xmin + g_xmax) / 2])
    l2_dis = K.sum((p_center - g_center)**2, axis=-1)
    diag_C = C_width**2 + C_height**2
    diou = iou - l2_dis / diag_C
    if iou_type == 'diou':
        return diou
    # v = [0.1, 0.1, 0.2, 0.2]      # experience
    arctan = tf.atan(g_width/g_height) - tf.atan(p_width/p_height)
    v = 4 * ((arctan / math.pi)**2)
    alpha = v / ((1 - iou) + v)
    ciou = iou - diou - alpha * v
    if iou_type == 'ciou':
        return ciou
    return iou













