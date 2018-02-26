#############################################
# author: Liu Zhengyuan
# date: 2018-01-11
# update: Attention-Based Sentiment Analysis
#############################################


import numpy as np
import codecs
import tensorflow as tf
from common.preprocess import load_w2v, load_aspect2id, load_inputs_twitter_with_aspect, generate_batch_index, generate_aspect_file
# from common.attention_visualization import batch_plot_attention
import os
import datetime
from models.dynamic_memory_network import DynamicMemoryNetwork
from models.fast_cnn import FastCNN
from models.conv_seq2seq import CnnSeq2Seq
from models.lstm_with_attention import AttentionLSTM


""" Valuables Configuration """
embedding_vector_len = 300

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float("learning_rate", 0.001, "the learning rate")
tf.app.flags.DEFINE_integer("batch_size", 100, "the batch size")

tf.app.flags.DEFINE_integer("embedding_vector_len", embedding_vector_len, "length of word embedding vector")
tf.app.flags.DEFINE_integer("max_sentence_len", 40, "length of the sentence")

tf.app.flags.DEFINE_integer("global_hidden_size", embedding_vector_len, "the embedding vector scale")
tf.app.flags.DEFINE_integer("num_classes", 3, "the number of labels")
tf.app.flags.DEFINE_integer("max_epochs", 5, "the training epochs")
tf.app.flags.DEFINE_float("global_dropout", 0.9, "the config of drop out")
tf.app.flags.DEFINE_float("l2_regularization", 0.002, "the config of l2 regularization")
tf.app.flags.DEFINE_boolean("is_training", True, "enable/disable the drop_out")


""" model option: DynamicMemoryNetwork/CnnSeq2Seq/AttentionLSTM/FastCNN"""
model_option = "FastCNN"

"""Advanced Configuration"""
"""
    "inference_hops": 2, "the number of inference hops"
    input_represent_opt: "input representation: Bidirectional_GRU/Forward_GRU/CNN"
    question_represent_opt: "question representation: RNN/Direct"
    attention_method: "the config of attention method: dot_product/tanh_concat"
    episode_option: "the config of episode calculation: Soft/LSTM/GRU"
    clip_gradients: True/False, "the config of clip_gradients, actually no need for now"
    "cnn_fusion", True/False, "the config of cnn fusion for episode calculation"
    "answer_opt": matrix_mul/full_connect, "the option of answer layer"
"""
DMN_params = {"inference_hops": 2,
              "input_represent_opt": "Forward_GRU",
              "question_represent_opt": "Direct",
              "attention_method": "tanh_concat",
              "episode_option": "GRU",
              "clip_gradients": False,
              "pos_embedding_opt": False,
              "answer_opt": "matrix_mul"}

CNN_params = {"inference_hops": 2,
              "input_represent_opt": "CNN",
              "question_represent_opt": "Direct",
              "attention_method": "tanh_concat",
              "episode_option": "Soft",
              "clip_gradients": False,
              "cnn_fusion": True,
              "pos_embedding_opt": False,
              "answer_opt": "matrix_mul"}


word_embedding_file = 'data/nuclear/Glove_embedding_file_nuclear_20180125.txt'

train_data_file = 'data/nuclear/train_tmp.txt'
test_data_file = 'data/nuclear/test_tmp.txt'
validate_data_file = ''


""" temporary writing new file for test """
merged_file = 'data/nuclear/merged_labeled_20180209.txt'
tmp_fp_merged = codecs.open(merged_file, "r", "utf-8").readlines()
tmp_fp_train = codecs.open(train_data_file, "w", "utf-8")
tmp_fp_test = codecs.open(test_data_file, "w", "utf-8")
if len(tmp_fp_merged) % 3 == 0:
    tmp_list = [x for x in range(int(len(tmp_fp_merged) / 3))]
    np.random.shuffle(tmp_list)
    tmp_split = int(len(tmp_list) * 0.9)

    tmp_write_list = []
    for i in tmp_list[:tmp_split]:
        for j in range(3):
            tmp_write_list.append(tmp_fp_merged[i * 3 + j])
    tmp_fp_train.writelines(tmp_write_list)
    tmp_write_list = []
    for i in tmp_list[tmp_split:]:
        for j in range(3):
            tmp_write_list.append(tmp_fp_merged[i * 3 + j])
    tmp_fp_test.writelines(tmp_write_list)
else:
    print("merged file rows error!")
    exit()


tmp_fp_test.close()
tmp_fp_train.close()
tmp_fp_merged = []


""""Read word2vec and generate aspect vector dictionary"""
word_id_mapping, w2v = load_w2v(word_embedding_file, FLAGS.embedding_vector_len)


class DataSetObject:
    def __init__(self, obj_name, word_id_map, word2vec, data_file, embedding_len, max_sen_len):
        self.name = obj_name
        self.data_file = data_file

        print(">>>>>Generate %s aspect id mapping file:" % self.name)
        self.aspect_id_file = generate_aspect_file(data_file, "data/"+self.name+"_aspect_id_new.txt")
        print(">>>>>Loading %s Data Set:" % self.name)
        self.aspect_id_mapping, self.aspect_vector = load_aspect2id(self.aspect_id_file,
                                                                    word_id_map,
                                                                    word2vec,
                                                                    embedding_len)
        self.x, self.sentence_len, self.target_word, self.y = \
            load_inputs_twitter_with_aspect(data_file, word_id_map, self.aspect_id_mapping, max_sen_len)
        print('Max %s sentence length of' % self.name, max(self.sentence_len), '\n')


train_data = DataSetObject("train_data", word_id_mapping, w2v,
                           train_data_file, FLAGS.embedding_vector_len, FLAGS.max_sentence_len)

test_data = DataSetObject("test_data", word_id_mapping, w2v,
                          test_data_file, FLAGS.embedding_vector_len, FLAGS.max_sentence_len)


print("batch size=", FLAGS.batch_size)


def get_batch_data(w2v, batch_idx, data_set, X, Y, Aspect_vector, Sequence_len):
    x = (tf.nn.embedding_lookup(w2v, np.asarray(data_set.x[batch_idx]))).eval()
    y = np.asarray(data_set.y[batch_idx])
    aspect_vec = (tf.nn.embedding_lookup(data_set.aspect_vector, data_set.target_word[batch_idx])).eval()
    seq_len = (data_set.sentence_len[batch_idx])
    return {X: x, Y: y, Aspect_vector: aspect_vec, Sequence_len: seq_len}


if model_option == "DynamicMemoryNetwork":
    model = DynamicMemoryNetwork(FLAGS.learning_rate, FLAGS.embedding_vector_len, FLAGS.max_sentence_len,
                                 FLAGS.global_hidden_size, FLAGS.num_classes, FLAGS.global_dropout,
                                 FLAGS.l2_regularization, FLAGS.is_training, DMN_params)

if model_option == "CnnSeq2Seq":
    model = CnnSeq2Seq(FLAGS.learning_rate, FLAGS.embedding_vector_len, FLAGS.max_sentence_len,
                       FLAGS.global_hidden_size, FLAGS.num_classes, FLAGS.global_dropout,
                       FLAGS.l2_regularization, FLAGS.is_training, CNN_params)

if model_option == "FastCNN":
    model = FastCNN(FLAGS.learning_rate, FLAGS.embedding_vector_len, FLAGS.max_sentence_len,
                    FLAGS.global_hidden_size, FLAGS.num_classes, FLAGS.global_dropout,
                    FLAGS.l2_regularization, FLAGS.is_training, CNN_params)

if model_option == "AttentionLSTM":
    model = AttentionLSTM(FLAGS.learning_rate, FLAGS.embedding_vector_len, FLAGS.max_sentence_len,
                          FLAGS.global_hidden_size, FLAGS.num_classes, FLAGS.global_dropout,
                          FLAGS.l2_regularization, FLAGS.is_training)


print("\nThis is %s !\n" % model.__class__.__name__)


""" Add tf train saver """
restore_train_model = False
save_train_model = False
save_interval = 5
saver = tf.train.Saver()
checkpoint_dir = "data\\tmp\\saver\\"


with tf.Session() as sess:
    sess.run(model.init)

    """ restore the train saver"""
    if restore_train_model is True:
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            print("###INFO### Restore train model from", checkpoint.model_checkpoint_path)
            saver.restore(sess, checkpoint.model_checkpoint_path)

    """ start train epoch """
    for epoch in range(1, FLAGS.max_epochs+1):

        epoch_train_start_time = datetime.datetime.now()

        """ Control the drop_out """
        model.is_training = True
        for batch_index in generate_batch_index(len(train_data.y), FLAGS.batch_size, num_iter=1, is_shuffle=True):
            batch_params = get_batch_data(w2v, batch_index, train_data,
                                          model.X, model.Y, model.Aspect_vector, model.Sequence_len)
            """ Train and show the result of training batch set """
            sess.run(model.train_op, feed_dict=batch_params)

        epoch_train_end_time = datetime.datetime.now()
        print("Epoch %d, Execute Time: " % epoch, (epoch_train_end_time - epoch_train_start_time).seconds)

        """ Calculate train epoch loss and accuracy """
        model.is_training = False
        train_eval_index = np.random.permutation(len(train_data.y))[0:len(test_data.y)]
        train_eval_feed = get_batch_data(w2v, train_eval_index, train_data,
                                         model.X, model.Y, model.Aspect_vector, model.Sequence_len)
        loss, acc = sess.run([model.loss_op, model.accuracy], feed_dict=train_eval_feed)
        print("Epoch " + str(epoch) + ", Epoch Loss= " + "{:.4f}".format(loss)
              + ", Training Accuracy= " + "{:.3f}".format(acc))

        """ Calculate test loss and accuracy """
        model.is_training = False
        test_index = range(len(test_data.y))
        test_feed = get_batch_data(w2v, test_index, test_data,
                                   model.X, model.Y, model.Aspect_vector, model.Sequence_len)
        loss, acc, pred_stat, attention_list = sess.run([model.loss_op, model.accuracy,
                                                        model.correct_pred, model.attention_output],
                                                        feed_dict=test_feed)

        print("Epoch " + str(epoch) + ", Test Loss= " + "{:.4f}".format(loss)
              + ", Test Accuracy= " + "{:.3f}".format(acc))

        """ Attention Visualization """
        if model_option == "DynamicMemoryNetwork":
            hop_tmp = DMN_params['inference_hops']
        elif model_option == "CnnSeq2Seq" or "FastCNN":
            hop_tmp = CNN_params['inference_hops']
        else:
            hop_tmp = 2

        print(np.shape(attention_list))
        for num_hop in range(hop_tmp):
            print(attention_list[num_hop][0][4])

        # plot_index = range(0, 50)
        # for num_hop in range(hop_tmp):
        #     plot_path = "data" + "\\tmp" + "\\epoch" + str(epoch) + "\\hop" + str(num_hop)
        #     if os.path.exists(plot_path) is not True:
        #         os.makedirs(plot_path)
        #     batch_plot_attention(hop=num_hop, attentions=attention_list,
        #                          source_file=test_data.data_file, index=plot_index,
        #                          epoch=epoch, path=plot_path,
        #                          predict=pred_stat)

        """ save the train model"""
        if (save_train_model is True) and (epoch % save_interval == 0):
            saver.save(sess, checkpoint_dir + 'model_saver.ckpt', global_step=epoch)

    print("Training iteration finished")


