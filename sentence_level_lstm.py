
import numpy as np
import tensorflow as tf
from common.process_function import load_w2v, load_aspect2id, load_inputs_twitter_with_aspect, generate_batch_index, generate_aspect_file, get_batch_data
import os
import datetime
from models.lstm import LSTM


""" Valuables Configuration """
embedding_vector_len = 300

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float("learning_rate", 0.001, "the learning rate")
tf.app.flags.DEFINE_integer("batch_size", 100, "the batch size")

tf.app.flags.DEFINE_integer("embedding_vector_len", embedding_vector_len, "length of word embedding vector")
tf.app.flags.DEFINE_integer("max_sentence_len", 110, "length of the sentence")

tf.app.flags.DEFINE_integer("global_hidden_size", embedding_vector_len, "the embedding vector scale")
tf.app.flags.DEFINE_integer("num_classes", 3, "the number of labels")
tf.app.flags.DEFINE_integer("max_epochs", 10, "the training epochs")
tf.app.flags.DEFINE_float("global_dropout", 0.9, "the config of drop out")
tf.app.flags.DEFINE_float("l2_regularization", 0.002, "the config of l2 regularization")
tf.app.flags.DEFINE_boolean("is_training", True, "enable/disable the drop_out")


""" model option: LSTM / GRU """
model_option = "LSTM"

word_embedding_file = './data/Glove_embedding_file_nuclear_20180413.txt'

train_data_file = './data/nuclear/train_tmp.txt'
test_data_file = './data/nuclear/test_tmp.txt'
validate_data_file = ''

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


if model_option == "LSTM":
    model = LSTM(FLAGS.learning_rate, FLAGS.embedding_vector_len, FLAGS.max_sentence_len,
                          FLAGS.global_hidden_size, FLAGS.num_classes, FLAGS.global_dropout,
                          FLAGS.l2_regularization, FLAGS.is_training)


print("\nThis is %s !\n" % model.__class__.__name__)


""" Add tf train saver """
restore_train_model = False
save_train_model = False
save_interval = 5
saver = tf.train.Saver()
checkpoint_dir = "./data/tmp/saver/"


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
        tmp_batch_num = 0
        for batch_index in generate_batch_index(len(train_data.y), FLAGS.batch_size, num_iter=1, is_shuffle=True):
            tmp_batch_num += 1
            if tmp_batch_num % 10 == 0 :
                print(tmp_batch_num)
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
        loss, acc, pred_stat = sess.run([model.loss_op, model.accuracy, model.correct_pred],
                                                        feed_dict=test_feed)

        print("Epoch " + str(epoch) + ", Test Loss= " + "{:.4f}".format(loss)
              + ", Test Accuracy= " + "{:.3f}".format(acc))

        """ save the train model"""
        if (save_train_model is True) and (epoch % save_interval == 0):
            saver.save(sess, checkpoint_dir + 'model_saver.ckpt', global_step=epoch)

    print("Training iteration finished")

