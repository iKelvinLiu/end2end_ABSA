import codecs
import numpy as np
import random
import re
import time

# """ 直接随机配对生成样本 """
# input_file = './data/nuclear/nuclear_dataset.txt'
# output_file = './data/new_sample/multi_sentence_data.txt'
#
# input_fp = codecs.open(input_file, "r", "utf-8").readlines()
# output_fp = codecs.open(output_file, "w", "utf-8")
#
# input_sample_num = int(len(input_fp) / 3)
#
# for i in range(input_sample_num):
#     random_idx = random.randint(0, input_sample_num-1)
#     while input_fp[i*3+1] == input_fp[random_idx*3+1] or input_fp[i*3+2] == input_fp[random_idx*3+2]:
#         random_idx = random.randint(0, input_sample_num-1)
#     tmp_sentence = input_fp[i*3].strip() + " . " + input_fp[random_idx*3]
#     tmp_sentence = re.sub(r"\s+", " ", tmp_sentence)
#     output_fp.writelines([tmp_sentence+"\n", input_fp[i*3+1], input_fp[i*3+2],
#                           tmp_sentence+"\n", input_fp[random_idx*3+1], input_fp[random_idx*3+2]])


""" 过滤掉具有相同词的句子进行配对 """
input_file = './data/nuclear/nuclear_dataset.txt'
output_file = './data/new_sample/multi_sentence_data.txt'

input_fp = codecs.open(input_file, "r", "utf-8").readlines()
output_fp = codecs.open(output_file, "w", "utf-8")

word_filter = {"nuclear cost": ["cost", "cheap", "expensive", "cheaper", "subsidy", "bills", "bill"],
               "nuclear efficiency": ["efficiency", "efficient"],
               "nuclear emission": ["emission", "carbon", "green", "clean"],
               "nuclear radiation": ["radiation", "radiative", "radioactive", "pollution", "poison", "polluted"],
               "nuclear safety": ["safety", "danger","dangerous", "unreliable", "safe", "risky", "hazardous", "disastrous", "reliable"],
               "nuclear waste": ["waste", "dump", "wastes", "dumps"],
               }

input_sample_num = int(len(input_fp) / 3)


def contain_aspect(word_filter, sentence):
    output = []
    for tmp_k in word_filter.keys():
        tmp = set(word_filter[tmp_k]) & set(sentence.split())
        if len(tmp) > 1:
            output.append(tmp_k)
    return output


def overlap_aspect(word_filter, sentence_1, sentence_2):
    output = set(contain_aspect(word_filter, sentence_1)) & set(contain_aspect(word_filter, sentence_2))
    if len(output) > 1:
        return True
    else:
        return False


for i in range(input_sample_num):
    random_idx = random.randint(0, input_sample_num-1)
    while input_fp[i*3+1] == input_fp[random_idx*3+1] or input_fp[i*3+2] == input_fp[random_idx*3+2] or \
            overlap_aspect(word_filter, input_fp[i*3], input_fp[random_idx*3]):
        random_idx = random.randint(0, input_sample_num-1)
    tmp_sentence = input_fp[i*3].strip() + " . " + input_fp[random_idx*3]
    tmp_sentence = re.sub(r"\s+", " ", tmp_sentence)
    output_fp.writelines([tmp_sentence+"\n", input_fp[i*3+1], input_fp[i*3+2],
                          tmp_sentence+"\n", input_fp[random_idx*3+1], input_fp[random_idx*3+2]])
