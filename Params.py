
import tensorflow as tf

# Data
tf.app.flags.DEFINE_string('input_training_data', 'data.txt', 'Training data path')
tf.app.flags.DEFINE_string('input_validation_data', 'test.txt', 'Validation data path')
tf.app.flags.DEFINE_string('input_prediction_data', 'pred.txt', 'Prediction data path')

# Mode
tf.app.flags.DEFINE_string('mode','train','train, predict or evaluation mode')
tf.app.flags.DEFINE_integer('feature_size', 1 << 29, 'Feature size.')

# Paramters
tf.app.flags.DEFINE_bool('self_custom_feature', False, "Whether using self-custom query-keyword cross funtion.")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_integer("traing_epochs", 20, "Training epochs.")
tf.app.flags.DEFINE_integer("batch_size", 512, "Batch size.")
tf.app.flags.DEFINE_integer("buffer_size", 1024, "Batch size.")
tf.app.flags.DEFINE_float("L1", 1, "L1 paramter.")
tf.app.flags.DEFINE_float("L2", 0, "L2 paramter.")
tf.app.flags.DEFINE_string("optimizer", "Grad", "Optimizer.")

tf.app.flags.DEFINE_integer('loss_cnt', 1, 'total loss count to update')
tf.app.flags.DEFINE_string('loss_mode', 'LogLoss', "Loss function Mode: LogLoss, MSE.")

tf.app.flags.DEFINE_string('output_model_path', 'model','path to save model')
tf.app.flags.DEFINE_string('log_dir','log', 'folder to save log')
tf.app.flags.DEFINE_integer('log_frequency', 1, 'log frequency during training procedure')
tf.app.flags.DEFINE_integer('checkpoint_frequency', 1, 'evaluation frequency during training procedure')

tf.app.flags.DEFINE_integer('max_model_to_keep', 10, 'max models to save')
tf.app.flags.DEFINE_string('result_filename','predict_res.txt','result file name')

FLAGS = tf.app.flags.FLAGS