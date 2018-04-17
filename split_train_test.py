import codecs
import numpy as np

merged_file = './data/nuclear/nuclear_dataset.txt'
train_data_file = './data/nuclear/train_tmp.txt'
test_data_file = './data/nuclear/test_tmp.txt'

merged_file = './data/new_sample/multi_sentence_data.txt'
train_data_file = './data/new_sample/train_tmp.txt'
test_data_file = './data/new_sample/test_tmp.txt'

tmp_fp_merged = codecs.open(merged_file, "r", "utf-8").readlines()
tmp_fp_train = codecs.open(train_data_file, "w", "utf-8")
tmp_fp_test = codecs.open(test_data_file, "w", "utf-8")

if len(tmp_fp_merged) % 3 == 0:
    tmp_list = [x for x in range(int(len(tmp_fp_merged) / 3))]
    print("All Sample Number: ", len(tmp_list))
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
