import keras.backend as K
import tensorflow as tf


def kp_loss(args, n_classes, n_points, scales=[8,16]):
    '''
    args: [*cls_outputs, *kp_outputs, *y_true], (b,h,w,(2+cls)*p)
    cls_outputs: for each grid, for each point, for each cls, exists or not
    kp_outputs: for each grid, for each point, normed-rela-grid-xy
    y_true: for each grid, for each point, normed-rela-origin-xy & cls vector
    returns: loss tensor, shape=(1,)
    '''
    num_levels = len(scales)
    cls_outputs = args[:num_levels]
    kp_outputs = args[num_levels:num_levels*2]
    y_true = args[num_levels*2:]
    grid_shapes = [K.int_shape(y_true[l])[1:3] for l in range(num_levels)]  # [[hs,ws], ...]

    loss = 0.
    m = K.shape(cls_outputs[0])[0]
    mf = K.cast(m, tf.float32)
    for l in range(num_levels):
        # grid
        grid_shape = grid_shapes[l]
        grid_y = K.tile(K.reshape(K.arange(0, grid_shape[0]), [-1, 1, 1]), [1, grid_shape[1], 1])  # tile row
        grid_x = K.tile(K.reshape(K.arange(0, grid_shape[1]), [1, -1, 1]), [grid_shape[0], 1, 1])  # tile column
        grid = K.concatenate([grid_x, grid_y], axis=-1)
        grid = K.cast(grid, tf.float32)

        # xy_loss
        grid = K.tile(grid, [1, 1, n_points])
        grid_shape_wh = K.cast(grid_shape[::-1], tf.float32)
        grid_shape_wh = K.tile(grid_shape_wh, n_points)
        gt_xy = y_true[l][..., :n_points*2]*grid_shape_wh - grid
        pred_xy = kp_outputs[l]
        true_mask = tf.where(y_true[l][..., :n_points*2]>0, tf.ones_like(pred_xy), tf.zeros_like(pred_xy))
        xy_loss = true_mask * K.binary_crossentropy(gt_xy, pred_xy, from_logits=True)

        # cls_loss
        gt_labels = y_true[l][..., n_points*2:]
        pred_labels = cls_outputs[l]
        cls_loss = K.binary_crossentropy(gt_labels, pred_labels, from_logits=True)
        cls_focal_loss = focal_loss(gt_labels, pred_labels, from_logits=True)

        # sum
        xy_loss = K.sum(xy_loss) / mf
        cls_loss = K.sum(cls_loss) / mf
        cls_focal_loss = K.sum(cls_focal_loss) / mf

        loss += xy_loss + cls_focal_loss
        loss = tf.Print(loss, [l, xy_loss, cls_loss, cls_focal_loss], message="   level,  xy_loss, cls_loss, cls_focal_loss: ")

    return loss


def focal_loss(y_true, y_pred, from_logits=True):
    gamma = 2.
    alpha = 0.25

    if from_logits:
        gap = K.abs(y_true - K.sigmoid(y_pred))
    else:
        gap = K.abs(y_true - y_pred)

    epsilon = K.epsilon()
    gap = K.clip(gap, epsilon, 1-epsilon)
    loss = - K.pow(gap, gamma) * K.log(1-gap)

    return loss








