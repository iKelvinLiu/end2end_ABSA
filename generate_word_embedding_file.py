import codecs
import re

abs_root = "./"


def load_w2v(w2v_file, embedding_dim):
    fp = codecs.open(w2v_file, 'r', 'utf-8')
    w2v = []
    word_dict = dict()
    cnt = 0
    flag = True

    while flag:
        line = fp.readline()
        if not line:
            flag = False
        else:
            line = str(line)
            line = line.split()
            if len(line) != embedding_dim + 1:
                print('a bad word embedding'+str(line))
                continue
            w2v.append([float(v) for v in line[1:]])
            print(str(line[0]))
            word_dict[str(line[0])] = cnt
            cnt += 1
    print('shape of new word2vec library:', len(w2v))
    return word_dict, w2v


def merge_txt_files(files, destination):
    fo = codecs.open(destination, 'w', 'utf-8')
    for name in files:
        fi = codecs.open(name, "r", 'utf-8')
        while True:
            content = fi.read()
            if not content:
                break
            fo.write(content)
        fi.close()
    fo.close()


def load_inputs_data(input_file):
    word_list = []
    lines = codecs.open(input_file, "r", 'utf-8').readlines()
    for i in range(0, len(lines)):
        words = lines[i].lower().split()
        for w in words:
            if w not in word_list:
                word_list.append(w)
    return word_list


def generate_new_embedding_file(word_list, word_dict, w2v, newfile_name):
    new_file = codecs.open(newfile_name, 'w', "'utf-8'")

    for w in word_list:
        if w in word_dict:
            tmp = str(w)
            for k in w2v[word_dict[w]]:
                tmp = tmp + " " + str(k)
            tmp = tmp + "\n"
            new_file.write(tmp)
    new_file.close()


file_list = [".//data/nuclear/nuclear_dataset.txt"]

merged_file = abs_root + "//embedding_files//new_merged_content.txt"
w2v_file = abs_root + "//embedding_files/glove.840B.300d.txt"
new_file_name = abs_root + "//embedding_files//new_embedding_file.txt"

embedding_dim = 300

merge_txt_files(file_list, merged_file)
print("Merge Finished")

word_list = load_inputs_data(merged_file)
print(len(word_list))

word_dict, w2v = load_w2v(w2v_file, embedding_dim)
print("Load w2v file Finished")

generate_new_embedding_file(word_list, word_dict, w2v, new_file_name)


"""do some filter after initial word2vec """
fp = codecs.open(new_file_name, 'r', 'utf-8')
fp_new = codecs.open(new_file_name+"filtered", 'w', 'utf-8')
lines = fp.readlines()
for i in lines:
    if (re.search("^[0-9][0-9]\:", i) is None) and (re.search("^[0-9]+\s", i) is None) \
            and (re.search("^[0-9]+\,[0-9]+\s", i) is None):
        fp_new.write(i)

