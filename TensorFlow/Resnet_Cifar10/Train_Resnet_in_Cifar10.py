import tensorflow as tf
from Data_Read import get_next_batch
from Resnet import resnet_model
import os
slim = tf.contrib.slim

def loss(logits, label):

    one_hot_label = slim.one_hot_encoding(label, 10)
    slim.losses.softmax_cross_entropy(logits, one_hot_label)

    reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    l2_loss = tf.add_n(reg_set)
    slim.losses.add_loss(l2_loss)

    totalloss = slim.losses.get_total_loss()

    return totalloss, l2_loss

def func_optimal(batchsize, loss_val):
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(0.01,
                                    global_step,
                                    decay_steps= 50000// batchsize,
                                    decay_rate= 0.95,
                                    staircase=False)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        op = tf.train.AdamOptimizer(lr).minimize(loss_val, global_step)
    return global_step, op, lr

def train():
    print("Start train!")
    batchsize = 32
    floder_log = r'TensorFlow\CNN_Model\logdir'
    floder_model = r'TensorFlow\CNN_Model\saved_model'

    if not os.path.exists(floder_log):
        os.mkdir(floder_log)

    if not os.path.exists(floder_model):
        os.mkdir(floder_model)

    tr_summary = set()
    #te_summary = set()

    # get data
    tr_im, tr_label = get_next_batch(batchsize, 0, 1)
    #te_im, te_label = get_next_batch(batchsize, 1, 0)

    ##net
    input_data = tf.placeholder(tf.float32, shape=[None, 32, 32, 3],
                                name='input_data')

    input_label = tf.placeholder(tf.int64, shape=[None],
                                name='input_label')
    keep_prob = tf.placeholder(tf.float32, shape=None,
                                name='keep_prob')

    is_training = tf.placeholder(tf.bool, shape=None,
                               name='is_training')
    logits = resnet_model(input_data, keep_prob=keep_prob, is_training=is_training)

    ##loss

    total_loss, l2_loss = loss(logits, input_label)

    tr_summary.add(tf.summary.scalar('train total loss', total_loss))
    tr_summary.add(tf.summary.scalar('test l2_loss', l2_loss))

    #te_summary.add(tf.summary.scalar('train total loss', total_loss))
    #te_summary.add(tf.summary.scalar('test l2_loss', l2_loss))

    ##accurancy
    pred_max  = tf.argmax(logits, 1)
    correct = tf.equal(pred_max, input_label)
    accurancy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tr_summary.add(tf.summary.scalar('train accurancy', accurancy))
    #te_summary.add(tf.summary.scalar('test accurancy', accurancy))
    ##op
    global_step, op, lr = func_optimal(batchsize, total_loss)
    tr_summary.add(tf.summary.scalar('train lr', lr))
    #te_summary.add(tf.summary.scalar('test lr', lr))

    tr_summary.add(tf.summary.image('train image', input_data * 128 + 128))
    #te_summary.add(tf.summary.image('test image', input_data * 128 + 128))

    with tf.Session() as sess:
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))

        tf.train.start_queue_runners(sess=sess,
                                     coord=tf.train.Coordinator())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        ckpt = tf.train.latest_checkpoint(floder_model)

        if ckpt:
            saver.restore(sess, ckpt)

        epoch_val = 100

        tr_summary_op = tf.summary.merge(list(tr_summary))
        #te_summary_op = tf.summary.merge(list(te_summary))

        summary_writer = tf.summary.FileWriter(floder_log, sess.graph)

        for i in range(50000 * epoch_val):
            train_im_batch, train_label_batch = \
                sess.run([tr_im, tr_label])
            feed_dict = {
                input_data:train_im_batch,
                input_label:train_label_batch,
                keep_prob:0.8,
                is_training:True
            }

            _, global_step_val, \
            lr_val, \
            total_loss_val, \
            accurancy_val, tr_summary_str = sess.run([op,
                                      global_step,
                                      lr,
                                      total_loss,
                                      accurancy, tr_summary_op],
                     feed_dict=feed_dict)

            summary_writer.add_summary(tr_summary_str, global_step_val)

            if i % 100 == 0:
                print("{},{},{},{}".format(global_step_val,
                                           lr_val, total_loss_val,
                                           accurancy_val))
            if i % 1000 == 0:
                saver.save(sess, "{}/model.ckpt{}".format(floder_model, str(global_step_val)))
    return

if __name__ == '__main__':
    train()