#####################################
# author: Liu Zhengyuan
# date: 2018-01-06
# update: preprocess of data
#####################################


import numpy as np
import codecs
import tensorflow as tf


def generate_aspect_file(input_file, new_file_name):
    new_file = open(new_file_name, 'w')
    word_list = []
    word_index = 1
    fp = codecs.open(input_file, 'r', 'utf-8')
    lines = fp.readlines()
    for i in range(0, len(lines), 3):
        aspect_word = ' '.join(lines[i + 1].lower().split())
        if aspect_word not in word_list:
            word_list.append(aspect_word)
            new_file.write(str(aspect_word) + " " + str(word_index) + '\n')
            word_index += 1
    new_file.close()
    fp.close()
    print(new_file_name)
    return new_file_name


def generate_batch_index(sample_size, batch_size, num_iter=1, is_shuffle=True):
    index = list(range(sample_size))
    for j in range(num_iter):
        if is_shuffle:
            np.random.shuffle(index)
        for i in range(int(sample_size / batch_size) + (1 if sample_size % batch_size else 0)):
            yield index[i * batch_size:(i + 1) * batch_size]


def change_y_to_onehot(y):
    """ 这个地方有点问题，之前每次输出的one hot编码会变化，现在先固定为3位hot编码 """
    # from collections import Counter
    # print(Counter(y))
    # class_set = set(y)
    # n_class = len(class_set)
    # y_onehot_mapping = dict(zip(class_set, range(n_class)))
    # onehot = []
    # for label in y:
    #     tmp = [0] * n_class
    #     tmp[y_onehot_mapping[label]] = 1
    #     onehot.append(tmp)
    # return np.asarray(onehot, dtype=np.int32)

    """ 现在先固定为3位hot编码 """
    one_hot = []
    for i in y:
        if i == '-1':
            one_hot.append([1, 0, 0])
        elif i == '0':
            one_hot.append([0, 1, 0])
        elif i == '1':
            one_hot.append([0, 0, 1])
        elif i == '3':
            one_hot.append([0, 1, 0])
        else:
            print('One_hot encoding error:', i)
    return np.asarray(one_hot, dtype=np.int32)


def load_w2v(w2v_file, embedding_dim, is_skip=False):
    fp = codecs.open(w2v_file, 'r', 'utf-8')
    if is_skip:
        fp.readline()
    w2v = []
    word_dict = dict()
    # [0,0,...,0] represent absent words
    w2v.append([0.] * embedding_dim)
    cnt = 0
    for line in fp:
        cnt += 1
        line = str(line)
        line = line.split()
        if len(line) != embedding_dim + 1:
            print('A bad word embedding'+str(line[0]))
            continue
        w2v.append([float(v) for v in line[1:]])
        word_dict[str(line[0])] = cnt
    w2v = np.asarray(w2v, dtype=np.float32)
    w2v = np.row_stack((w2v, np.sum(w2v, axis=0) / cnt))
    print('Shape of new word2vec library:', np.shape(w2v))
    word_dict['$t$'] = (cnt + 1)
    # w2v -= np.mean(w2v, axis=0)
    # w2v /= np.std(w2v, axis=0)
    print('Word "$t$" vector index:', word_dict['$t$'], 'New word2vec dict:', len(w2v))
    fp.close()
    return word_dict, w2v


def load_inputs_twitter(input_file, word_id_file, max_sentence_len):

    word_to_id = word_id_file
    x, y, sen_len = [], [], []
    target_words = []
    lines = open(input_file).readlines()
    for i in range(0, len(lines), 3):
        target_word = lines[i + 1].lower().split()
        target_word = list(map(lambda w: word_to_id.get(w, 0), target_word))
        target_words.append([target_word[0]])

        y.append(lines[i + 2].strip().split()[0])

        words = lines[i].lower().split()
        words_l, words_r = [], []
        flag = True
        for word in words:
            if word == '$t$':
                flag = False
                continue
            if flag:
                if word in word_to_id:
                    words_l.append(word_to_id[word])
            else:
                if word in word_to_id:
                    words_r.append(word_to_id[word])
        words = words_l + target_word + words_r
        sen_len.append(len(words))
        x.append(words + [0] * (max_sentence_len - len(words)))

    y = change_y_to_onehot(y)
    return np.asarray(x), np.asarray(sen_len), np.asarray(y)


def load_inputs_twitter_with_aspect(input_file, word_id_file, aspect_id_file, max_sentence_len):

    word_to_id = word_id_file
    aspect_to_id = aspect_id_file

    x, y, sen_len = [], [], []
    aspect_words = []
    fp = codecs.open(input_file, 'r', 'utf-8')
    lines = fp.readlines()
    if len(lines) % 3 != 0:
        print("Error lines alignment")
        exit()
    for i in range(0, len(lines), 3):
        aspect_word = ' '.join(lines[i + 1].lower().split())
        aspect_words.append(aspect_to_id.get(aspect_word, 0))

        y.append(lines[i + 2].split()[0])

        words = lines[i].lower().split()
        ids = []
        for word in words:
            if word in word_to_id:
                ids.append(word_to_id[word])
            else:
                # print("cannot find %s in word2vec" % word)
                ids.append(0)
        # ids = list(map(lambda word: word_to_id.get(word, 0), words))
        sen_len.append(len(ids))
        x.append(ids + [0] * (max_sentence_len - len(ids)))
    cnt = 0
    for item in aspect_words:
        if item > 0:
            cnt += 1
    print('Valid Target Word Num=' + str(cnt) + ", Sample Size=" + str(len(y)))
    print(y[0:10])
    y = change_y_to_onehot(y)
    for item in x:
        if len(item) != max_sentence_len:
            print('#Error# with max_sentence_length=', len(item), "\n#Solution# Change the max_sentence_len with a larger number")
            exit()
    fp.close()
    return x, np.asarray(sen_len), aspect_words, np.asarray(y)


def load_aspect2id(input_file, word_id_mapping, w2v, embedding_dim):
    aspect2id = dict()
    a2v = list()
    a2v.append([0.] * embedding_dim)
    cnt = 0
    temp_invalid_item = 0
    for line in open(input_file):
        line = line.lower().split()
        cnt += 1
        aspect2id[' '.join(line[:-1])] = cnt
        tmp = []
        for word in line:
            if word in word_id_mapping and len(word) > 1:
                tmp.append(w2v[word_id_mapping[word]])
        if tmp:
            a2v.append(np.sum(tmp, axis=0) / len(tmp))
        else:
            #print("Random Generate the aspect vector:", line)
            temp_invalid_item += 1
            a2v.append(np.random.uniform(-0.01, 0.01, (embedding_dim,)))

    print("Number of target words:", len(aspect2id), "Length of target word2vec:", len(a2v))
    print("Random Generate the aspect vector:", temp_invalid_item)
    return aspect2id, np.asarray(a2v, dtype=np.float32)


def get_batch_data(w2v, batch_idx, data_set, X, Y, Aspect_vector, Sequence_len):
    tmp_x = []
    for idx in batch_idx:
        tmp_item = []
        for item in data_set.x[idx]:
            tmp_item.append(w2v[item])
        tmp_x.append(tmp_item)
    tmp_x = np.asarray(tmp_x)

    y = np.asarray(data_set.y[batch_idx])

    tmp_aspect_vec = []
    for idx in batch_idx:
        tmp_aspect_vec.append(data_set.aspect_vector[data_set.target_word[idx]])
    tmp_aspect_vec = np.asarray(tmp_aspect_vec)

    # print(np.shape(tmp_x), np.shape(tmp_aspect_vec))
    seq_len = (data_set.sentence_len[batch_idx])
    return {X: tmp_x, Y: y, Aspect_vector: tmp_aspect_vec, Sequence_len: seq_len}

