### efficientNet
    tf官方仓库：https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
    keras实现：https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
    activation: 基本上conv-bn-activation组合中的激活函数都是swish，除了SE-block和最后概率输出的'sigmoid'和'softmax'
    initializer: efficientNet自定义了CONV_KERNEL_INITIALIZER和DENSE_KERNEL_INITIALIZER，抄过来了
    residual: inverse_DW_block的PW输出是没有激活函数的，每个efficientBlock最后add了id path以后，也不添加activation
    se_ratio: se-block里面的reduced dim是filters_in*se_ratio，而不是expand_dim*se_ratio
    b0: Total params: 5,250,196
    b1-b7: depth & width & resolution 同比增加

    

### efficientDet
    tf官方仓库: https://github.com/google/automl/tree/master/efficientdet
    det_model_fn.py: 一些training details
    efficientdet_arch.py: 模型定义

    learning rate: Learning rate is proportional to the batch size, make sense的，因为batch size越大越接近真实分布

    loss: cls_loss->focal_loss, box_regression_loss->huber_loss, box_iou_loss, reg_l2_loss

    features for FPN: level3-7，也就是8x-128x下采样，但是efficientNet-b0的output stride只有32，
    原代码中针对不存在对应尺度的feature：Adds a coarser level by downsampling the last feature map, 也可能是要上采样，
    源代码中下采样到对应尺度用pooling，上采样用nearest neighbor

    biFPN: 所有input_node统一1x1conv-bn调整通道数，pooling／resize调整尺寸(resample_feature_map)，然后weighted add

    fuse_features: fusion来自不同节点的feature的时候，如果求加权和会定义一个list of tensor，里面的tensor都是标量，
    keras的层运算默认第一维是batch dim，只能在2d及以上的tensor上运行，但是tfbackend的softmax、sum等操作可以作用于标量
    stack&unstack: 常量数组和一维向量之间的转换

    cls&box head: 不同尺度的特征图复用，几个conv-bn-swish+id的block，然后加conv head

    Lambda wrapper: keras骚操作





