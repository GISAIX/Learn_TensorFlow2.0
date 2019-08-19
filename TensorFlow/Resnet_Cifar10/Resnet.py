import tensorflow as tf 
slim = tf.contrib.slim

def resnet_block(net,output_num,down,stride,is_training):
    '''
    Resnet block.

    #Arguments:
    net:input.
    output_num:output channels.
    down
    stride
    is_traing
    '''
    batch_norm_params = {
        'is_training':is_training,
        'decay':0.997,
        'epsilon':1e-5,
        'scale':True,
        'updates_collections':tf.GraphKeys.UPDATE_OPS
    }

    #slim.arg_scope中可以定义一些函数的默认参数值
    #在scope内重复用到这些函数时可以不用把所有参数都写一遍。
    with slim.arg_scope(
        [slim.conv2d],
        weights_regularizer = slim.l2_regularizer(0.0001),#权重正则化器
        weights_initializer = slim.variance_scaling_initializer(),#权重初始化器
        activation_fn = tf.nn.relu,#激活函数
        normalizer_fn = slim.batch_norm,#正则化函数
        normalizer_params = batch_norm_params): 
        with slim.arg_scope([slim.conv2d,slim.max_pool2d],padding='SAME') as arg_sc:

            shortcut = net
            
            if output_num != net.get_shape().as_list()[-1]:
                shortcut = slim.conv2d(net,output_num,[1,1])
            
            if stride != 1:
                shortcut = slim.max_pool2d(shortcut,[3,3],stride=stride)

            net = slim.conv2d(net,output_num//down,[1,1])
            net = slim.conv2d(net,output_num//down,[3,3])
            net = slim.conv2d(net,output_num,[1,1])

            if stride!=1:
                net = slim.max_pool2d(net,[3,3],stride=stride)

            net = net+shortcut
            return net

def resnet_model(net,keep_prob=0.5,is_training=True):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME') as arg_sc:

        net = slim.conv2d(net, 64, [3, 3], activation_fn=tf.nn.relu)
        net = slim.conv2d(net, 64, [3, 3], activation_fn=tf.nn.relu)

        net = resnet_block(net, 128, 4, 2, is_training)
        net = resnet_block(net, 128, 4, 1, is_training)
        net = resnet_block(net, 256, 4, 2, is_training)
        net = resnet_block(net, 256, 4, 1, is_training)
        net = resnet_block(net, 512, 4, 2, is_training)
        net = resnet_block(net, 512, 4, 1, is_training)

        net = tf.reduce_mean(net, [1, 2])
        net = slim.flatten(net)

        net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='fc1')
        net = slim.dropout(net, keep_prob, scope='dropout1')
        net = slim.fully_connected(net, 10, activation_fn=None, scope='fc2')

    return net
