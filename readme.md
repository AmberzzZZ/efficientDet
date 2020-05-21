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
