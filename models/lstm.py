import tensorflow as tf
import numpy as np

class LSTM(object):
    def __init__(self, learning_rate, embedding_vector_len, max_sentence_len, global_hidden_size,
                 num_classes, global_dropout, l2_regularization, is_training):

        """ Basic Configuration """
        self.learning_rate = learning_rate
        self.embedding_vector_len = embedding_vector_len
        self.max_sentence_len = max_sentence_len
        self.global_hidden_size = global_hidden_size
        self.num_classes = num_classes
        self.global_dropout = global_dropout
        self.l2_regularization = l2_regularization
        self.is_training = is_training

        """Placeholder Configuration"""
        self.X = tf.placeholder("float", [None, self.max_sentence_len, self.embedding_vector_len])
        self.Y = tf.placeholder("float", [None, self.num_classes])
        self.Sequence_len = tf.placeholder(tf.int32, [None])
        self.Aspect_vector = tf.placeholder("float", [None, self.embedding_vector_len])

        """for sess run"""
        self.prob = self.main_process(self.X, self.Sequence_len)
        self.loss_op = self.calculate_loss_op(self.Y, self.prob)
        self.train_op = self.training_op(self.loss_op)

        self.correct_pred = tf.equal(tf.argmax(tf.nn.softmax(self.prob), 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        """ global_variables_initializer """
        self.init = tf.global_variables_initializer()

    def main_process(self, X, Sequence_len):
        with tf.variable_scope("main_process", initializer=tf.contrib.layers.xavier_initializer()):
            with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):
                _, state = tf.nn.dynamic_rnn(tf.nn.rnn_cell.BasicLSTMCell(self.global_hidden_size), X,
                                           dtype=np.float32, sequence_length=Sequence_len)
                print(state)
            with tf.variable_scope('dense', reuse=tf.AUTO_REUSE):
                output = tf.layers.dense(state[1], 3)
                print(output)
        return output

    def calculate_loss_op(self, Y, prob):
        l2 = self.l2_regularization
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.cast(Y, tf.float32), logits=prob))
        for v in tf.trainable_variables():
            if 'bias' not in v.name.lower():
                print("adding l2 regularization", v.name.lower())
                loss += l2 * tf.nn.l2_loss(v)
        return loss

    def training_op(self, loss_value):
        max_grad_val = 5.0
        """Calculate and apply gradients"""
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gvs = opt.compute_gradients(loss_value)

        train_output = opt.apply_gradients(gvs)

        return train_output



