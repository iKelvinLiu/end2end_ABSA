import numpy as np
import tensorflow as tf
from models.attention_gru_cell import AttentionGRUCell
from models.conv_encoder import conv_encode


class DynamicMemoryNetwork(object):
    def __init__(self, learning_rate, embedding_vector_len, max_sentence_len, global_hidden_size,
                 num_classes, global_dropout, l2_regularization, is_training, param_dict):

        """ Basic Configuration """
        self.learning_rate = learning_rate
        self.embedding_vector_len = embedding_vector_len
        self.max_sentence_len = max_sentence_len
        self.global_hidden_size = global_hidden_size
        self.num_classes = num_classes
        self.global_dropout = global_dropout
        self.l2_regularization = l2_regularization
        self.is_training = is_training

        """Advanced Configuration"""
        self.inference_hops = param_dict["inference_hops"]
        self.input_represent_opt = param_dict["input_represent_opt"]
        self.question_represent_opt = param_dict["question_represent_opt"]
        self.attention_method = param_dict["attention_method"]
        self.episode_option = param_dict["episode_option"]
        self.clip_gradients = param_dict["clip_gradients"]
        self.pos_embedding_opt = param_dict["pos_embedding_opt"]
        self.answer_opt = param_dict["answer_opt"]

        """Placeholder Configuration"""
        self.X = tf.placeholder("float", [None, self.max_sentence_len, self.global_hidden_size])
        self.Y = tf.placeholder("float", [None, self.num_classes])
        self.Sequence_len = tf.placeholder(tf.int32, [None])
        self.Aspect_vector = tf.placeholder("float", [None, self.global_hidden_size])

        """for sess run"""
        self.prob, self.attention_output = self.inference(self.X, self.Aspect_vector, self.Sequence_len)
        self.loss_op = self.calculate_loss_op(self.Y, self.prob)
        self.train_op = self.training_op(self.loss_op)

        self.correct_pred = tf.equal(tf.argmax(tf.nn.softmax(self.prob), 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        """ global_variables_initializer """
        self.init = tf.global_variables_initializer()

    def get_input_representation(self, inputs, seqlen):
        with tf.variable_scope("input_representation", reuse=tf.AUTO_REUSE):
            """ Convolution seq2seq for input representation """
            if self.input_represent_opt == "CNN":
                if self.pos_embedding_opt is True:
                    position_embedding = tf.get_variable(name="pos_embedding",
                                                         shape=[self.max_sentence_len, self.global_hidden_size],
                                                         initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
                else:
                    position_embedding = None
                cnn_output = conv_encode(inputs, seqlen, position_embedding, self.is_training, self.pos_embedding_opt)
                """ outputs shape=(?, ?, 300) dtype=float32 """
                fact_vecs = tf.reshape(cnn_output.outputs, [-1, self.max_sentence_len, self.global_hidden_size])
                return fact_vecs

            """ Memory_set为word embedding转化之后的vector集合，context """
            if self.input_represent_opt == "Bidirectional_GRU":
                """Get fact (sentence) vectors via embedding, positional encoding and bi-directional GRU"""
                forward_gru_cell = tf.contrib.rnn.GRUCell(self.global_hidden_size)
                backward_gru_cell = tf.contrib.rnn.GRUCell(self.global_hidden_size)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(forward_gru_cell,
                                                             backward_gru_cell,
                                                             inputs,
                                                             dtype=np.float32,
                                                             sequence_length=seqlen
                                                             )
                """ sum forward and backward output vectors """
                fact_vecs = tf.reduce_sum(tf.stack(outputs), axis=0)
                """ apply dropout """
                fact_vecs = tf.contrib.layers.dropout(fact_vecs, keep_prob=self.global_dropout, is_training=self.is_training)

                """ shape=(?, sentence_len, embedding_number) 对于每一个样本生成hidden size的internal representation """
                return fact_vecs

            if self.input_represent_opt == "Forward_GRU":
                """ Forward GRU """
                forward_gru_cell = tf.contrib.rnn.GRUCell(self.global_hidden_size)
                outputs, _ = tf.nn.dynamic_rnn(forward_gru_cell,
                                               inputs,
                                               dtype=np.float32,
                                               sequence_length=seqlen
                                               )
                fact_vecs = outputs
                """ apply dropout """
                fact_vecs = tf.contrib.layers.dropout(fact_vecs, keep_prob=self.global_dropout, is_training=self.is_training)

                """ shape=(?, sentence_len, embedding_number) 对于每一个样本生成hidden size的internal representation """
                return fact_vecs
            if self.input_represent_opt == "Direct":
                return inputs

    def get_question_representation(self, target_word):
        with tf.variable_scope("question_representation", reuse=tf.AUTO_REUSE):
            if self.question_represent_opt == "RNN":
                """ 需要提供question的word embedding和sequence length """
                """ Tensor("DMN/question/rnn/while/Exit_2:0", shape=(100, 80), dtype=float32) """
                hidden_size = self.global_hidden_size
                questions = tf.expand_dims(target_word, 1)
                gru_cell = tf.contrib.rnn.GRUCell(hidden_size)
                _, q_vec = tf.nn.dynamic_rnn(gru_cell,
                                             questions,
                                             dtype=np.float32
                                             )
                return q_vec

            if self.question_represent_opt == "Direct":
                """ shape=(?, 100) 按照word embedding直接输出 """
                q_vec = target_word
                return q_vec

    def get_attention(self, prev_memory, fact_vec):
        with tf.variable_scope("attention_mechanism", reuse=tf.AUTO_REUSE):
            """Use question vector and previous memory to create scalar attention for current fact"""
            if self.attention_method == "tanh_concat":
                hidden_size = self.global_hidden_size

                feature_vec = tf.concat([prev_memory, fact_vec], 1)

                attention = tf.contrib.layers.fully_connected(feature_vec,
                                                              hidden_size,
                                                              activation_fn=tf.nn.tanh,
                                                              reuse=tf.AUTO_REUSE, scope="fc1")

                attention = tf.contrib.layers.fully_connected(attention,
                                                              1,
                                                              activation_fn=None,
                                                              reuse=tf.AUTO_REUSE, scope="fc2")
                # attention = 0.5 * (1 + 0.2) * attention + 0.5 * (1 - 0.2) * abs(attention)

                return attention

            if self.attention_method == "dot_product":
                """ fact_vec // q_vec shape = (?, 300)  attention: shape (?, 1)"""
                attention = tf.multiply(fact_vec, prev_memory)
                attention = tf.reduce_sum(attention, 1, keepdims=True)
                return attention

    def generate_episode(self, memory, fact_vecs, seqlen):
        """Generate episode by applying attention to current fact vectors """
        hidden_size = self.global_hidden_size
        epi_attn_list = []

        """ Calculate each attention of facts. """
        attentions = [tf.squeeze(self.get_attention(memory, fv), axis=1)
                      for i, fv in enumerate(tf.unstack(fact_vecs, axis=1))]

        attentions = tf.transpose(tf.stack(attentions))

        """ Adding mask to the softmax layer """
        """ Replace all attention values for padded inputs with tf.float32.min """
        max_len = tf.shape(attentions)[1]
        adding_mask = tf.sequence_mask(lengths=tf.to_int32(seqlen),
                                       maxlen=tf.to_int32(max_len),
                                       dtype=tf.float32)
        attentions = attentions * adding_mask + ((1.0 - adding_mask) * tf.float32.min)

        """ Once we have the attention gate with softmax, 我们用这个attention来提取contextual vector c_t """
        attentions = tf.nn.softmax(attentions)

        epi_attn_list.append(attentions)

        attentions = tf.expand_dims(attentions, axis=-1)

        if self.episode_option == "Soft":
            """ DMN plus论文中提到的Soft Attention """
            """ TensorFlow的multiply支持broadcasting """
            with tf.variable_scope('Soft', reuse=tf.AUTO_REUSE):
                episode = tf.multiply(attentions, fact_vecs)
                episode = tf.reduce_sum(episode, 1)
                #print(episode)

        if self.episode_option == "LSTM":
            """ Using conventional LSTM """
            """ concatenate fact vectors and attentions for input into attGRU """
            inputs = tf.concat([fact_vecs, attentions], 2)

            with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):
                _, episode = tf.nn.dynamic_rnn(tf.nn.rnn_cell.BasicLSTMCell(hidden_size),
                                               inputs,
                                               dtype=np.float32,
                                               sequence_length=seqlen)
                episode = episode[0]

        if self.episode_option == "GRU":
            """ DMN plus论文中提到的AttentionGRU """
            """ concatenate fact vectors and attentions for input into attGRU """
            gru_inputs = tf.concat([fact_vecs, attentions], 2)

            with tf.variable_scope('attention_gru', reuse=tf.AUTO_REUSE):
                _, episode = tf.nn.dynamic_rnn(AttentionGRUCell(hidden_size),
                                               gru_inputs,
                                               dtype=np.float32,
                                               sequence_length=seqlen)

        return episode, epi_attn_list

    def inference(self, X, Aspect_vector, Sequence_len):
        """ Performs inference on the DMN model """
        num_hops = self.inference_hops
        hidden_size = self.global_hidden_size

        with tf.variable_scope("question", initializer=tf.contrib.layers.xavier_initializer()):
            q_vec = self.get_question_representation(Aspect_vector)

        with tf.variable_scope("input", initializer=tf.contrib.layers.xavier_initializer()):
            fact_vecs = self.get_input_representation(X, Sequence_len)

        infer_attn_list = []

        """ memory module """
        with tf.variable_scope("memory", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> build episodic memory')

            """ generate n_hops episodes """
            prev_memory = q_vec

            for i in range(num_hops):
                """ get a new episode """
                print('==> generating episode', i)
                episode, attn_tmp = self.generate_episode(prev_memory, fact_vecs, Sequence_len)
                infer_attn_list.append(attn_tmp)
                """ untied weights for memory update """
                with tf.variable_scope("hop_%d" % i):
                    prev_memory = tf.layers.dense(tf.concat([episode, q_vec], 1), hidden_size)
            output = prev_memory

        """ pass memory module output through linear answer module """
        with tf.variable_scope("answer", initializer=tf.contrib.layers.xavier_initializer()):
            output = self.add_answer_module(output, q_vec)

        return output, infer_attn_list

    def add_answer_module(self, rnn_output, q_vec):
        with tf.variable_scope("classification", reuse=tf.AUTO_REUSE):
            """ Linear softmax answer module """
            drop_out_placeholder = self.global_dropout
            rnn_output = tf.contrib.layers.dropout(rnn_output, keep_prob=drop_out_placeholder, is_training=self.is_training)

            if self.answer_opt == "full_connect":
                """ Using dense layer to control dimension """
                # output = tf.layers.dense(tf.concat([rnn_output, q_vec], 1), self.num_classes, activation=None)
                output = tf.layers.dense(rnn_output, self.num_classes, activation=None)

            if self.answer_opt == "matrix_mul":
                """ initialize the weights """
                Weight = tf.get_variable("Weight", [self.embedding_vector_len, self.num_classes])
                bias = tf.get_variable("bias", [self.num_classes])
                """ Using matrix multiply to control dimension """
                output = tf.matmul(rnn_output, Weight) + bias

            return output

    def calculate_loss_op(self, Y, prob):
        l2 = self.l2_regularization
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.cast(Y, tf.float32), logits=prob))
        for v in tf.trainable_variables():
            # print(v.name.lower())
            if 'bias' not in v.name.lower():
                print("adding l2 regularization", v.name.lower())
                loss += l2 * tf.nn.l2_loss(v)
        return loss

    def training_op(self, loss_value):
        max_grad_val = 5.0
        """Calculate and apply gradients"""
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gvs = opt.compute_gradients(loss_value)

        # optionally clip gradients to regularize
        if self.clip_gradients:
            gvs = [(tf.clip_by_norm(grad, max_grad_val), var) for grad, var in gvs]

        train_output = opt.apply_gradients(gvs)
        return train_output

