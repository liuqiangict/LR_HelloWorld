
import os
import sys
import time

import numpy as np
import tensorflow as tf

from tensorflow.python.client import device_lib
from datetime import datetime
from Params import FLAGS


class Trainer:
    def __init__(self, model, inc_step, train_reader, eval_reader = None, infer_reader = None):
        self.model = model

        self.inc_step = inc_step
        self.reader = train_reader
        self.eval_reader = eval_reader
        self.infer_reader = infer_reader

        self.devices = self.get_devices()
        self.total_weight = tf.Variable(0., trainable=False)
        self.total_loss = tf.Variable(0., trainable=False)
                
        opt = self.model.get_optimizer()

        tower_grads = []
        tower_loss = []

        self.weight_record = 0
        tf.get_variable_scope().reuse_variables()

        if FLAGS.mode == 'train':
            for i in range(0, len(self.devices)):
                with tf.device(self.devices[i]):
                    with tf.name_scope('Device_%d' % i) as scope:
                        batch_input = self.reader.get_next()
                        loss, weight = self.tower_loss(batch_input)
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        grads = []
                        grads.extend(opt.compute_gradients(loss))
                        tower_grads.append(grads)
                        tower_loss.append((loss, weight))
            self.avg_loss = [self.update_loss(tower_loss[i], i) for i in range(0, len(tower_loss))]
            grads = self.sum_gradients(tower_grads)
            self.train_op = opt.apply_gradients(grads)

        if FLAGS.mode == 'predict':
            tower_infer = []
            for i in range(0, len(self.devices)):
                with tf.device(self.devices[i]):
                    infer_batch = self.reader.get_next()
                    infer_res = self.tower_inference(infer_batch)
                    tower_infer.append([infer_batch, infer_res])
            self.infer_list = self.merge_infer_res(tower_infer)
        pass

    def tower_loss(self, batch_input):
        inference_output = self.model.inference(batch_input,tf.contrib.learn.ModeKeys.TRAIN)
        loss, weight = self.model.calc_loss(inference_output)
        tf.summary.scalar("Losses", loss)
        return loss, weight

    def update_loss(self, tower_loss, idx):
        loss, weight = tower_loss
        loss_inc = tf.assign_add(self.total_loss, tf.reduce_sum(loss))
        weight_inc = tf.assign_add(self.total_weight, tf.cast(tf.reduce_sum(weight), tf.float32))
        avg_loss = loss_inc / weight_inc * FLAGS.batch_size
        tf.summary.scalar("Avg_loss" + str(idx), avg_loss)
        return avg_loss

    def sum_gradients(self, tower_grads):
       sum_grads = []
       print(tower_grads)
       for grad_and_vars in zip(*tower_grads):
           print(grad_and_vars)
           if isinstance(grad_and_vars[0][0], tf.Tensor):
               grads = []
               for g, _ in grad_and_vars:
                   expanded_g = tf.expand_dims(g,0)
                   grads.append(expanded_g)
               print(grads)
               grad = tf.concat(grads, 0)
               grad = tf.reduce_sum(grad, 0)
               v = grad_and_vars[0][1]
               grad_and_var = (grad, v)
               sum_grads.append(grad_and_var)
           else:
               values = tf.concat([g.values for g,_ in grad_and_vars],0)
               indices = tf.concat([g.indices for g,_ in grad_and_vars],0)
               v = grad_and_vars[0][1]
               grad_and_var = (tf.IndexedSlices(values, indices),v)
               sum_grads.append(grad_and_var)
       return sum_grads

    def get_devices(self):
        #local_device_protos = device_lib.list_local_devices()
        #devices = [x.name for x in local_device_protos if x.device_type == 'GPU']
        devices = []
        if not len(devices):
            devices.append('/cpu:0')
        print("available devices", devices)
        return devices

    
    def train_ops(self):
        return [self.train_op, self.avg_loss, self.total_weight, self.inc_step]

    def print_log(self, total_weight, step, avg_loss):
        examples, self.weight_record = total_weight - self.weight_record, total_weight
        current_time = time.time()
        duration, self.start_time = current_time - self.start_time, time.time()
        examples_per_sec = examples * 10000 / duration
        sec_per_steps = float(duration / FLAGS.log_frequency)
        format_str = "%s: step %d, %5.1f examples/sec, %.4f sec/step, %f samples processed, "
        avgloss_str = "avg_loss = " + ",".join([str(avg_loss[i]) for i in range(0, len(avg_loss))])
        print(format_str % (datetime.now(), step, examples_per_sec, sec_per_steps, total_weight) + avgloss_str)
        pass

    def tower_inference(self, batch_input):
        inference_output = self.model.inference(batch_input, tf.contrib.learn.ModeKeys.INFER)
        a, b, prediction = self.model.predict(inference_output)
        return a, b, prediction

    def merge_infer_res(self, tower_infer):
        infer_batch, infer_res = zip(*tower_infer)
        merge_batch = []
        merge_res = []
        for i in zip(*infer_batch):
            if not isinstance(i[0],tf.Tensor):
                merge_batch.append(tf.concat([j[0] for j in i], axis = 0))
            else:
                merge_batch.append(tf.concat(i, axis=0))
        merge_res.append(tf.concat(infer_res[0][2], axis=0))
        return merge_batch, merge_res

    def predict(self, sess, mode, outputter):
        assert(mode == tf.contrib.learn.ModeKeys.INFER)
        while True:
            try:
                input_batch, score = sess.run(self.infer_list)
                for i in range(len(score)):
                    output_str = input_batch[0][i].decode("utf-8") + "\t" + input_batch[1][i].decode("utf-8") + "\t" + input_batch[2][i].decode("utf-8")  + "\t"
                    output_str += str(score[0][i])
                    outputter.write(output_str + "\n")
            except tf.errors.OutOfRangeError:
                print("score predict done.")
                break