### efficientNet
    tf官方仓库：https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
    keras实现：https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py

    activation: 基本上conv-bn-activation组合中的激活函数都是swish，除了SE-block和最后概率输出的'sigmoid'和'softmax'

    initializer: efficientNet自定义了CONV_KERNEL_INITIALIZER和DENSE_KERNEL_INITIALIZER，抄过来了

    residual: inverse_DW_block的PW输出是没有激活函数的，每个efficientBlock最后add了id path以后，也不添加activation

    se_ratio: se-block里面的reduced dim是filters_in*se_ratio，而不是expand_dim*se_ratio

    b0: Total params: 5,250,196
    b1-b7: depth & width & resolution 同比增加

    dropout & drop connect:
    dropout_rate: float, dropout rate before final classifier layer.
    dropout参数noise_shape: 第一次使用到这个参数，noise_shape的和输入tensor_shape的dim一致或者为1，
    例如本例可以是(None,h,w,c)、(None,h,w,1)等，哪个轴为1，哪个轴就会被一致地dropout
    drop_connect_rate: float, dropout rate at skip connections, fraction of the input units to drop
    源代码的dropconnect: 用在skip connection的地方，x+id，我们随机抛弃掉部分残差特征，然后add，跟原始定义有点区别
    原始的dropout: 用在全连接层，将节点输出以1-p的概率变成0
    原始的dropconnect: 用在全连接层，将节点中的每个与其相连的输入权值以1-p的概率变成0


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
    tf.add_n([p1, p2, p3....]): 可以实现一个列表的元素的相加

    cls&box head: 不同尺度的特征图复用，几个conv-bn-swish+id的block，然后加conv head，
    源代码裸的输出（没加损失函数，没做概率／loc的归一化），在loss里面有处理

    Lambda wrapper: keras骚操作

    cls loss: focal loss = (1-p_t)^r * log(pt)
    cls activation是sigmoid, 正样本pt_1=sigmoid(x), 负样本pt_0=1-sigmoid(x)
    unstable issue: r<1时bp会unstable，loss前半段(1-p_t)^r，用x替换log(e^x)
    loss后半段log(pt)，就是sigmoid cross entropy
    normalize item: RetinaNet里面提到的，focal loss是sum over all ～100k anchors and normalized by the number of anchors assigned to a ground-truth box

    box reg loss: huber loss
    reg output是linear output，没有activation
    误差较小时，接近MSE，梯度会随着损失值接近其最小值逐渐减少
    误差较大时，接近MAE，对异常值更敏感，梯度大
    coord order: [y_min, x_min, y_max, x_max]*, 而不是yolo的[x,y,w,h]
    valid_mask: 跟yolo一样y_true里面有box的vector写box coord，没有box的vector全0

    前背景：retinanet里面定义了anchor state {-1:ignore, 0:negative, 1:positive}，iou在[0.4,0.5]区间内的anchor计算cls loss时忽略，
    efficientDet里面只分前背景，clsloss全部背景类都参与运算


### box version & kp version & polygon version & (mask version)
    对于一阶段检测框架，可变的模块如下：
    1. back & fpn：看个人喜好就好
    2. shared／individual head：有些框架头上的repeats conv是share的，有些是独立的
    3. last layer：输出conv层，由具体任务决定，
        feature scale
        anchor／anchor-free: FCOS
        activation
        box／kp／polygon: poly-yolo
    4. loss：跟last layer和任务关联


### kp version
    基于回归的关键点定位，是检测框的简化版本，
    1. 不需要anchor，因为没有宽高
    2. 结构化输出，类别数cls，点的个数p(固定／上限)
        一个grid内只有一个点: (B,H,W,2+cls)，grid size要通过先验统计确定好
        一个grid内可以有多个点: (B,H,W,(2+cls)*p)，一个极限情况是global average pooling的特征图，一个grid，全图回归
    3. kp_loss: 就是box_xy_loss + cls_loss
        xy预测的是normed-rela-grid-xy, [0,1]之间，可以用bce，只在gt上有点的grid上计算
        cls预测的是当前grid中有无当前类别的点，[0,1]之间，可以用bce／focal loss，在box上是对positive和negative计算，
        （但是对于关键点，落进／没落进格子没那么ambiguous，可以全图算，应该不需要ignore）





















