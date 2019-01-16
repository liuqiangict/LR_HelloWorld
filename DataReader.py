
import sys
import tensorflow as tf
from CrossFeatures_SelfCustom import CrossFeatures_SelfCustom

class DataReader():
    def __init__(self, filename, buffer_size, batch_size, num_epochs, is_shuffle=True):
        self.crossFeaturer = CrossFeatures_SelfCustom()
        dataset = tf.data.TextLineDataset(filename)
        dataset = dataset.map(self.parse_line, num_parallel_calls = 1)
        if is_shuffle:
            dataset = dataset.shuffle(buffer_size = buffer_size)
        dataset = dataset.repeat(num_epochs)
        dataset_batch = dataset.batch(batch_size)
        dataset_batch = dataset_batch.prefetch(buffer_size=buffer_size)
        self.iterator = dataset_batch.make_initializable_iterator()

    def parse_line(self, line):
        columns = tf.decode_csv(line, [[""] for i in range(0, 3)], field_delim="\t", use_quote_delim=False)
        res = self.convert(columns)
        return res

    def convert(self, columns):
        res = tf.py_func(self.crossFeaturer.GetCrossFeature, [columns[0], columns[1]], [tf.string])
        res = [columns[0], columns[1]] + res +  [tf.string_to_number(columns[2], tf.float32)]
        return res

    def get_next(self):
        return self.iterator.get_next()

if __name__ == '__main__':
    reader = DataReader('data.txt', 1, 1, 1, False)
    with tf.Session() as sess:
        sess.run(reader.iterator.initializer)
        for i in range(0,9):
            feature = reader.get_next()
            #print(sess.run(x))
            print(sess.run(feature))
