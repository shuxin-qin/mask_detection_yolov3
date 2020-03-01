# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import cfgs
import time
import os

from utils.dataset import Dataset
from model import yolov3
from utils.eval_utils import evaluate_on_cpu

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

# setting placeholders
is_training = tf.placeholder(tf.bool, name="phase_train")
input_images = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='input_images')
input_labels_13 = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3, None], name='input_labels_13')
input_labels_26 = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3, None], name='input_labels_26')
input_labels_52 = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3, None], name='input_labels_52')

yolo_model = yolov3(cfgs.class_num, cfgs.anchors, cfgs.use_label_smooth, cfgs.use_focal_loss, 
                    cfgs.batch_norm_decay, cfgs.weight_decay, use_static_shape=False)

with tf.variable_scope('yolov3'):
    pred_feature_maps = yolo_model.forward(input_images, is_training=is_training)
    y_pred = yolo_model.predict(pred_feature_maps)
    loss = yolo_model.compute_loss(pred_feature_maps, [input_labels_13, input_labels_26, input_labels_52])

l2_loss = tf.losses.get_regularization_loss()

# setting restore parts and vars to update
saver_to_restore = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(
                                  include=cfgs.restore_include, exclude=cfgs.restore_exclude))
update_vars = tf.contrib.framework.get_variables_to_restore(include=cfgs.update_part)

global_step = tf.Variable(float(cfgs.global_step), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
learning_rate = tf.train.exponential_decay(1e-5, global_step, decay_steps=200, decay_rate=0.99, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate)

tf.summary.scalar('train_batch_statistics/total_loss', loss[0])
tf.summary.scalar('train_batch_statistics/loss_xy', loss[1])
tf.summary.scalar('train_batch_statistics/loss_wh', loss[2])
tf.summary.scalar('train_batch_statistics/loss_conf', loss[3])
tf.summary.scalar('train_batch_statistics/loss_class', loss[4])
tf.summary.scalar('train_batch_statistics/loss_l2', l2_loss)
tf.summary.scalar('learning_rate', learning_rate)

saver_to_save = tf.train.Saver(max_to_keep=2)

# set dependencies for BN ops
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    #train_op = optimizer.minimize(loss[0] + l2_loss, var_list=update_vars, global_step=global_step)
    gvs = optimizer.compute_gradients(loss[0] + l2_loss, var_list=update_vars)
    clip_grad_var = [gv if gv[0] is None else [tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in gvs]
    train_op = optimizer.apply_gradients(clip_grad_var, global_step=global_step)

train_dataset = Dataset(cfgs.train_path, cfgs.classes, cfgs.img_size, cfgs.anchors, 'train', batch_size=cfgs.batch_size, multi_scale=True)
val_dataset = Dataset(cfgs.val_path, cfgs.classes, cfgs.img_size, cfgs.anchors, 'test', batch_size=cfgs.batch_size, multi_scale=False)

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    saver_to_restore.restore(sess, cfgs.restore_path)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(cfgs.log_dir, graph=sess.graph)

    print('Trainable Variables: ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    print('Training Variables: ', np.sum([np.prod(v.get_shape().as_list()) for v in update_vars]))

    print('\n----------- start to train -----------\n')

    for epoch in range(cfgs.total_epoches):

        for data in train_dataset:
            
            start = time.time()
            _, imgs, y_true_13, y_true_26, y_true_52 = data
            _, summary, _y_pred, _loss, _global_step, _lr = sess.run([train_op, merged, y_pred, loss, global_step, learning_rate],
                feed_dict={is_training:True, input_images:imgs, input_labels_13:y_true_13, input_labels_26:y_true_26, input_labels_52:y_true_52})
            times = time.time()-start

            writer.add_summary(summary, global_step=_global_step)
            loss_total, loss_xy, loss_wh, loss_conf, loss_class = _loss[0], _loss[1], _loss[2], _loss[3], _loss[4]
            
            recall, precision = evaluate_on_cpu(_y_pred, [y_true_13, y_true_26, y_true_52], cfgs.class_num)

            if _global_step % cfgs.train_evaluation_step == 0 and _global_step > 0:

                info = "Epoch:{}, step:{}, time:{:.2f} | loss: total:{:.2f}, xy:{:.2f}, wh:{:.2f}, conf:{:.2f}, class:{:.2f}, lr:{:.6f}, R:{:.2f}, P:{:.2f}".format(
                        epoch, int(_global_step), times, loss_total, loss_xy, loss_wh, loss_conf, loss_class, _lr, recall, precision)
                print(info)

        if (epoch+1) % cfgs.val_evaluation_epoch == 0:

            for v_data in val_dataset:
                _, imgs, y_true_13, y_true_26, y_true_52 = v_data
                _y_pred, _loss = sess.run([y_pred, loss], 
                    feed_dict={is_training:False, input_images:imgs, input_labels_13:y_true_13, input_labels_26:y_true_26, input_labels_52:y_true_52})

                loss_total, loss_xy, loss_wh, loss_conf, loss_class = _loss[0], _loss[1], _loss[2], _loss[3], _loss[4]
                recall, precision = evaluate_on_cpu(_y_pred, [y_true_13, y_true_26, y_true_52], cfgs.class_num)

                info = "Eval:{}, | loss: total:{:.2f}, xy:{:.2f}, wh:{:.2f}, conf:{:.2f}, class:{:.2f}, R:{:.2f}, P:{:.2f}".format(
                        epoch, int(_global_step), times, loss_total, loss_xy, loss_wh, loss_conf, loss_class, recall, precision)
                print(info)
                
                break

        if (epoch+1) % cfgs.save_epoch == 0:
            save_name = os.path.join(cfgs.save_dir, 'model_' + str(epoch) + '.ckpt')
            saver_to_save.save(sess, save_name)
            print('save:', save_name)


