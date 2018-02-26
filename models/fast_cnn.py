import tensorflow as tf
from models.conv_encoder import conv_encode


class FastCNN(object):
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
        self.attention_method = param_dict["attention_method"]
        self.clip_gradients = param_dict["clip_gradients"]
        self.pos_embedding_opt = param_dict["pos_embedding_opt"]
        self.answer_opt = param_dict["answer_opt"]

        """Placeholder Configuration"""
        self.X = tf.placeholder("float", [None, self.max_sentence_len, self.embedding_vector_len])
        self.Y = tf.placeholder("float", [None, self.num_classes])
        self.Sequence_len = tf.placeholder(tf.int32, [None])
        self.Aspect_vector = tf.placeholder("float", [None, self.embedding_vector_len])

        """for sess run"""
        self.prob, self.attention_output = self.main_process(self.X, self.Aspect_vector, self.Sequence_len)
        self.loss_op = self.calculate_loss_op(self.Y, self.prob)
        self.train_op = self.training_op(self.loss_op)

        self.correct_pred = tf.equal(tf.argmax(tf.nn.softmax(self.prob), 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        """ global_variables_initializer """
        self.init = tf.global_variables_initializer()

    def get_input_representation(self, inputs, seqlen):
        """ Convolution seq2seq for input representation """
        if self.pos_embedding_opt is True:
            with tf.variable_scope("input_representation", reuse=tf.AUTO_REUSE):
                position_embedding = tf.get_variable(name="pos_embedding",
                                                     shape=[self.max_sentence_len, self.global_hidden_size],
                                                     initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        else:
            position_embedding = None
        cnn_output = conv_encode(inputs, seqlen, position_embedding, self.is_training, self.pos_embedding_opt)
        """ outputs shape=(?, ?, 300) dtype=float32 """
        fact_vecs = tf.reshape(cnn_output.outputs, [-1, self.max_sentence_len, self.global_hidden_size])
        return fact_vecs

    def get_question_representation(self, target_word):
        """ shape=(?, 100) 按照word embedding直接输出 """
        q_vec = target_word
        return q_vec

    def attention_layer(self, fact_vec, target_vec):
        """Use question vector and  target vector to create scalar attention """
        if self.attention_method == "tanh_concat":
            with tf.variable_scope("attention", reuse=tf.AUTO_REUSE):
                features = [fact_vec, target_vec]
                feature_vec = tf.concat(features, 1)

                attention = tf.contrib.layers.fully_connected(feature_vec,
                                                              self.global_hidden_size,
                                                              activation_fn=tf.nn.tanh,
                                                              reuse=tf.AUTO_REUSE, scope="fc1")
                attention = tf.contrib.layers.fully_connected(attention,
                                                              1,
                                                              activation_fn=None,
                                                              reuse=tf.AUTO_REUSE, scope="fc2")
            return attention

        if self.attention_method == "dot_product":
            """ fact_vec // q_vec shape = (?, 300)  attention: shape (?, 1)"""
            attention = tf.multiply(fact_vec, target_vec)
            attention = tf.reduce_sum(attention, 1, keep_dims=True)
            return attention

    def generate_inference(self, fact_vec, q_vec, seqlen):
        """Generate inference by applying attention to fact vectors """
        infer_attn_list = []

        """ Calculate each attention of facts. """
        attentions = [tf.squeeze(self.attention_layer(fv, q_vec), axis=1)
                      for i, fv in enumerate(tf.unstack(fact_vec, axis=1))]

        attentions = tf.transpose(tf.stack(attentions))

        """ Adding mask to the softmax layer """
        """ Replace all attention values for padded inputs with tf.float32.min """
        max_len = self.max_sentence_len
        adding_mask = tf.sequence_mask(lengths=tf.to_int32(seqlen), maxlen=tf.to_int32(max_len), dtype=tf.float32)
        attentions = attentions * adding_mask + ((1.0 - adding_mask) * tf.float32.min)

        """ Once we have the attention gate with softmax, combine attention with contextual vector """
        attentions = tf.nn.softmax(attentions)

        infer_attn_list.append(attentions)

        attentions = tf.expand_dims(attentions, axis=-1)

        episode = tf.multiply(attentions, fact_vec)
        episode = tf.reduce_sum(episode, 1)

        concat_infer = True
        if concat_infer is True:
            with tf.variable_scope("concatenate", initializer=tf.contrib.layers.xavier_initializer()):
                """episode shape=(?, 300), q_vec shape=(?, 300) """
                infer = tf.layers.dense(tf.concat([episode, q_vec], 1),
                                        self.global_hidden_size, activation=tf.nn.relu)
        else:
            infer = episode

        return infer, infer_attn_list

    def answer_module(self, infer_output):
        """ Linear softmax answer module """
        drop_out_placeholder = self.global_dropout
        infer_output = tf.contrib.layers.dropout(infer_output, keep_prob=drop_out_placeholder,
                                                 is_training=self.is_training)

        if self.answer_opt == "full_connect":
            """ Using dense layer to control dimension """
            output = tf.layers.dense(infer_output, self.num_classes, activation=None)

        if self.answer_opt == "matrix_mul":
            """ initialize the weights """
            with tf.variable_scope("classification",
                                   reuse=tf.AUTO_REUSE,
                                   initializer=tf.contrib.layers.xavier_initializer()):
                Weight = tf.get_variable("Weight", [self.embedding_vector_len, self.num_classes])
                bias = tf.get_variable("bias", [self.num_classes])
            """ Using matrix multiply to control dimension """
            output = tf.matmul(infer_output, Weight) + bias

        return output

    def main_process(self, X, Aspect_vector, Sequence_len):
        """ initiate attention list """
        attention_list = []

        with tf.variable_scope("question", initializer=tf.contrib.layers.xavier_initializer()):
            q_vec = self.get_question_representation(Aspect_vector)

        with tf.variable_scope("input", initializer=tf.contrib.layers.xavier_initializer()):
            fact_vecs = self.get_input_representation(X, Sequence_len)

        with tf.variable_scope("inference", initializer=tf.contrib.layers.xavier_initializer()):
            pre_infer = q_vec
            for hop in range(self.inference_hops):
                with tf.variable_scope("hop%d" % hop, initializer=tf.contrib.layers.xavier_initializer()):
                    pre_infer, attn_tmp = self.generate_inference(fact_vecs, pre_infer, Sequence_len)
                attention_list.append(attn_tmp)

        with tf.variable_scope("answer", initializer=tf.contrib.layers.xavier_initializer()):
            output = self.answer_module(pre_infer)

        return output, attention_list

    def calculate_loss_op(self, Y, prob):
        l2 = self.l2_regularization
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.cast(Y, tf.float32), logits=prob))
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

        # optionally clip gradients to regularize
        if self.clip_gradients:
            gvs = [(tf.clip_by_norm(grad, max_grad_val), var) for grad, var in gvs]

        train_output = opt.apply_gradients(gvs)

        return train_output



