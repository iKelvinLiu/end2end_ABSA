import tensorflow as tf
from tensorflow.contrib import rnn


class AttentionLSTM(object):
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

        self.X = tf.placeholder("float", [None, self.max_sentence_len, self.global_hidden_size])
        self.Y = tf.placeholder("float", [None, self.num_classes])
        self.Sequence_len = tf.placeholder(tf.int32, [None])
        self.Aspect_vector = tf.placeholder("float", [None, self.global_hidden_size])

        self.weights = {'softmax': tf.Variable(tf.random_normal([self.global_hidden_size, self.num_classes]))}
        self.biases = {'softmax': tf.Variable(tf.random_normal([self.num_classes]))}

        # TODO: rename these 4 weights for better readability
        self.W = tf.get_variable(
            name='W',
            shape=[self.global_hidden_size + self.embedding_vector_len,
                   self.global_hidden_size + self.embedding_vector_len],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
        )
        self.w = tf.get_variable(
            name='w',
            shape=[self.global_hidden_size + self.embedding_vector_len, 1],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
        )
        self.Wp = tf.get_variable(
            name='Wp',
            shape=[self.global_hidden_size, self.global_hidden_size],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
        )
        self.Wx = tf.get_variable(
            name='Wx',
            shape=[self.global_hidden_size, self.global_hidden_size],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
        )

        """ learning process """
        self.prob, self.attention_output = self.attention_lstm(self.X, self.Aspect_vector, self.Sequence_len,
                                                               self.weights, self.biases,
                                                               self.W, self.w, self.Wp, self.Wx,
                                                               self.is_training)
        self.loss_op = self.calculate_loss_op(self.Y, self.prob)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_op)

        self.correct_pred = tf.equal(tf.argmax(self.prob, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        """ variables initializer """
        self.init = tf.global_variables_initializer()

    @staticmethod
    def softmax_layer(inputs, weights, biases, keep_prob, is_training):
        outputs = tf.contrib.layers.dropout(inputs, keep_prob=keep_prob, is_training=is_training)
        predict = tf.matmul(outputs, weights) + biases
        predict = tf.nn.softmax(predict)
        return predict

    @staticmethod
    def reduce_mean(inputs, length):
        """
        :param inputs: 3-D tensor
        :param length: the length of dim [1]
        :return: 2-D tensor
        """
        length = tf.cast(tf.reshape(length, [-1, 1]), tf.float32) + 1e-9
        inputs = tf.reduce_sum(inputs, 1, keep_dims=False) / length
        return inputs

    @staticmethod
    def softmax(inputs, length, max_length):
        inputs = tf.cast(inputs, tf.float32)
        max_axis = tf.reduce_max(inputs, 2, keep_dims=True)
        inputs = tf.exp(inputs - max_axis)
        length = tf.reshape(length, [-1])
        mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_length), tf.float32), tf.shape(inputs))
        inputs *= mask
        _sum = tf.reduce_sum(inputs, reduction_indices=2, keep_dims=True) + 1e-9
        return inputs / _sum

    def attention_lstm(self, inputs, target, seq_len, weights, biases, W, w, Wp, Wx, is_training):
        batch_size = tf.shape(inputs)[0]
        target = tf.reshape(target, [-1, 1, self.embedding_vector_len])
        target = tf.ones([batch_size, self.max_sentence_len, self.embedding_vector_len], dtype=tf.float32) * target
        in_t = tf.concat([inputs, target], 2)
        in_t = tf.contrib.layers.dropout(in_t, keep_prob=self.global_dropout, is_training=is_training)

        cell = rnn.BasicLSTMCell(self.global_hidden_size)
        hiddens, _state = tf.nn.dynamic_rnn(cell, inputs=in_t, sequence_length=seq_len, dtype=tf.float32)
        # print(hiddens)

        h_t = tf.reshape(tf.concat([hiddens, target], 2), [-1, self.global_hidden_size + self.global_hidden_size])
        M = tf.matmul(tf.tanh(tf.matmul(h_t, W)), w)

        """ here is the attention score """
        alpha = self.softmax(tf.reshape(M, [-1, 1, self.max_sentence_len]), seq_len, self.max_sentence_len)

        """ calculate the output with attention score"""
        r = tf.reshape(tf.matmul(alpha, hiddens), [-1, self.global_hidden_size])
        index = tf.range(0, batch_size) * self.max_sentence_len + (seq_len - 1)
        hn = tf.gather(tf.reshape(hiddens, [-1, self.global_hidden_size]), index)
        h = tf.tanh(tf.matmul(r, Wp) + tf.matmul(hn, Wx))

        """ output for attention visualization """
        attn_ouput = tf.expand_dims(tf.expand_dims(tf.reshape(alpha, [-1, self.max_sentence_len]), axis=0), axis=0)

        return self.softmax_layer(h, weights['softmax'], biases['softmax'],
                                  keep_prob=self.global_dropout,
                                  is_training=self.is_training), attn_ouput

    def calculate_loss_op(self, Y, prob):
        l2 = self.l2_regularization
        basic_loss = - tf.reduce_mean(tf.cast(Y, tf.float32) * tf.log(prob))
        # TODO """ add l2 will make the attention alpha no changes """
        # for v in tf.trainable_variables():
        #     if 'bias' not in v.name.lower():
        #         print("adding l2 regularization", v.name.lower())
        #         basic_loss += l2 * tf.nn.l2_loss(v)
        return basic_loss

