# total_loss: cls_loss, box_loss, box_iou_loss
import tensorflow as tf
import keras.backend as K
import math


def det_loss(args, config):
    # args: *cls_outputs, *box_outputs, *y_true, from x8 to x128
    # cls_outputs: [b,h,w,a*cls], logits for each class
    # box_outputs: [b,h,w,a*4], rela-grid-xy & rela-anchor-wh
    # y_true: [b,h,w,a*(4+cls)], rela-origin-xywh & logits
    n_levels = len(args)//3
    cls_preds = args[:n_levels]
    box_preds = args[n_levels:2*n_levels]
    box_gts = args[2*n_levels:2*n_levels+4]
    cls_gts = args[2*n_levels+4:]

    total_cls_loss = 0.
    total_box_reg_loss = 0.
    total_box_iou_loss = 0.
    for i in range(n_levels):
        # convert
        box_gt = origin2grid(box_gts[i], config['aspect_ratios'], config['num_scales'])
        box_pred_xy = K.sigmoid(box_preds[i][...,:2])     # normed rela-grid-xy
        box_pred_wh = K.exp(box_preds[i][...,2:])     # rela-anchor-wh
        box_pred = tf.concat([box_pred_xy, box_pred_wh], axis=-1)
        # cls loss
        cls_loss = focal_loss(cls_preds[i], cls_gts[i], config['alpha'], config['gamma'])
        total_cls_loss += cls_loss
        # reg loss
        box_reg_loss = reg_loss(box_pred, box_gt, config['delta'])
        total_box_reg_loss += box_reg_loss
        # iou loss
        box_iou_loss = iou_loss(box_pred, box_gt, config['iou_loss_type'])
        total_box_iou_loss += box_iou_loss

        total_box_iou_loss = tf.Print(total_box_iou_loss,
                                      [i, cls_loss, box_reg_loss, box_iou_loss],
                                      message=" level & cls & reg & iou loss: ")

    total_loss = total_cls_loss + config['box_loss_weight']*total_box_reg_loss + config['iou_loss_weight']*total_box_iou_loss

    total_loss = tf.Print(total_loss, [total_cls_loss, total_box_reg_loss, total_box_iou_loss],
                          message=" total cls & reg & iou loss: ")

    return total_loss


def focal_loss(cls_pred, cls_true, alpha=0.25, gamma=2.0):
    # focal loss: -(1-p_t)^r * log(pt)
    # # ------- google implemention
    # pos_mask = tf.equal(cls_true, 1.0)
    # # first half: stable form, exp(-r*z*x - r*log(1+exp(-x)))
    # first_half = K.exp(gamma*cls_true*(-cls_pred) - gamma*K.log(1+K.exp(-cls_pred)))
    # # second half: log(pt), sigmoid bce
    # second_half = tf.where(pos_mask, -K.log(K.sigmoid(cls_pred)), -K.log(1-K.sigmoid(cls_pred)))
    # loss = first_half * second_half
    # focal_loss = tf.where(pos_mask, alpha*loss, (1-alpha)*loss)
    # ------- raw implemention
    cls_pred = K.sigmoid(cls_pred)
    pt = 1 - K.abs(cls_pred - cls_true)
    focal_loss = -K.pow(1-pt, gamma) * K.log(pt)
    focal_loss = tf.where(cls_true>0, (1-alpha)*focal_loss, alpha*focal_loss)
    norm_term = K.sum(cls_true, axis=[1,2,3])
    focal_loss = K.sum(focal_loss, axis=[1,2,3]) / norm_term
    return K.mean(focal_loss)


def reg_loss(box_pred, box_true, delta=0.1):
    # huber loss: mse & mae
    valid_mask = tf.not_equal(box_true, 0.0)
    huber_loss = tf.where((K.abs(box_true-box_pred)<delta),
                          0.5*((box_true-box_pred)**2),
                          delta*K.abs(box_true-box_pred) - 0.5*(delta**2))
    huber_loss = K.sum(huber_loss * valid_mask, axis=[1,2,3])
    return K.mean(huber_loss)


def iou_loss(box_pred, box_true, iou_type):
    box_pred_list = tf.unstack(box_pred, None, axis=-1)    # unstack 4A-axis
    box_true_list = tf.unstack(box_true, None, axis=-1)
    iou_list = []
    # tranverse xywh for each anchor
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


def origin2grid(box_gt, anchor_ratio, anchor_scales):
    # box_gt: [B,H,W,4A]
    # convert normed-rela-origin-xy to normed-rela-grid-xy
    # convert normed-rela-origin-wh to rela-anchor-wh


    return box_gt







