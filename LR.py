
import sys
import tensorflow as tf

from Params import FLAGS


class LRModel():
    def __init__(self):
        self.W = tf.get_variable(name='Weights', shape=[FLAGS.feature_size], dtype=tf.float32, initializer=tf.random_normal_initializer())
        self.b = tf.get_variable(name='Bisas', shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer) 

        self.l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.005, scope=None)
        pass

    def inference(self, input_fields, mode):
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            str_features, str_labels = input_fields[2], input_fields[3]
        elif mode == tf.contrib.learn.ModeKeys.INFER:
            str_features, str_labels = input_fields[2], "1"

        features = tf.string_split(str_features, ';')
        features = tf.SparseTensor(
            indices = features.indices,
            values=tf.string_to_number(features.values, out_type=tf.int32),
            dense_shape=features.dense_shape)
        labels = tf.reshape(str_labels, [-1])
            
        product = tf.nn.embedding_lookup_sparse(self.W, features, None, combiner='sum')
        pred = tf.nn.sigmoid(product + self.b)
            
        return input_fields[0], input_fields[1], labels, pred

    def calc_loss(self, inference_res):
        a, b, labels, pred = inference_res
        batch_size = tf.shape(pred)[0]
        
        if(FLAGS.loss_mode == 'LogLoss'):
            unweighted_loss = tf.losses.log_loss(labels=labels, predictions=pred)
        elif(FLAGS.loss_mode == 'MSE'):
            unweighted_loss = tf.losses.mean_squared_error(labels=labels, predictions=pred)

        loss = tf.reduce_sum(unweighted_loss)
        if FLAGS.L1 is not None:
            l1_regularization_penalty = tf.contrib.layers.apply_regularization(self.l1_regularizer, tf.trainable_variables())
            loss = tf.add(loss,  FLAGS.L1 * l1_regularization_penalty)
        if FLAGS.L2 is not None:
            l2_regularizer = tf.nn.l2_loss(self.W)
            loss = tf.add(loss, FLAGS.L2 * tf.reduce_mean(l2_regularizer))

        weight = batch_size

        return loss, weight

    def predict(self, inference_res):
        a, b, labels, pred = inference_res
        return a, b, pred

    def get_optimizer(self, optimizer_mode = "Grad"):
        if optimizer_mode == "Adam":
            return tf.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate)
        elif optimizer_mode == "Grad":
            return tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
        elif optimizer_mode == "FTRL":
            return tf.train.FtrlOptimizer(learning_rate=FLAGS.learning_rate, l1_regularization_strength=FLAGS.L1, l2_regularization_strength=FLAGS.L2)