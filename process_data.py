from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
import json
from itertools import chain
import pandas as pd
import numpy as np
import pickle
from os import makedirs
from os.path import exists, join
from keras.preprocessing.sequence import pad_sequences


def saveData():
    file = open('annotations.txt', 'r', encoding='UTF-8')
    js = file.read();
    dic = json.loads(js);
    dict1 = dic["annotationsOfQueries"]

    # 去除停用词
    stopwords = []
    file = open("stopwords.txt", "r")
    for line in file:  # 设置文件对象并读取每一行文件
        line = line.strip('\n')
        stopwords.append(line)

    # 存入文件
    train_x_file = open('data.txt', 'a')
    train_y_file = open("label.txt", 'a')

    for lv in dict1:
        str_x = ''
        str_y = ''
        for word in lv["query"].split():
            # if word not in stopwords:
            flag = 0;
            for an in lv["annotations"]:
                if word == an["value"]:
                    flag = 1
                    if an["category"] == "Data Format":  # category可能是data format，词组不行！！！！！！
                        str_y += "Data-Format" + " "
                    elif an["category"] == "Other Entities":
                        str_y += "Other-Entities" + " "
                    elif an["category"] == "Other Numbers":
                        str_y += "Other-Numbers" + " "
                    else:
                        str_y += an["category"] + " "
                    str_x += word + " "
                    break;
            if flag == 0:
                str_y += "other" + " "
                str_x += word + " "

        str_y.strip('')
        str_x.strip('')
        str_x += "\n"
        str_y += "\n"

        train_x_file.write(str_x)
        train_y_file.write(str_y)


def process_data(data, vocab, maxlen=20):
    print("process_data  data:"), print(data)

    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [word2idx.get(w[0].lower(), 1) for w in data]
    print("x:"), print(x)

    length = len(x)
    print("process_data length:"), print(length)

    x = pad_sequences([x], maxlen)  # left padding
    print("__x:"), print(x)

    return x, length


datafile = open('data.txt', 'r')
labelfile = open('label.txt', 'r')

words, labels = [], []

count = 0
for data, label in zip(datafile, labelfile):
    count += 1
    s1 = data.strip('\n').split(' ')
    s2 = label.strip('\n').split(' ')

    words.append(s1)
    labels.append(s2)

# print(words)
# print(labels)

datafile.close()
labelfile.close()

# 拆分出训练集测试集

# Get words set
all_words = list(chain(*words))  # words为二维数组，通过chain和*，将words拆成一维数组
# print(all_words)
all_words_sr = pd.Series(all_words)  # 序列化 类似于一维数组的对象，它由一组数据（各种NumPy数据类型）以及一组与之相关的数据标签（即索引）组成。
# print(all_words_sr)
all_words_counts = all_words_sr.value_counts()  # 计算字频
# print(all_words_counts)
all_words_set = all_words_counts.index  # index为字，values为字的频数，降序
# print(all_words_set)

# Get words ids
all_words_ids = range(1, len(all_words_set) + 1)  # 字典，从1开始

# Dict to transform
word2id = pd.Series(all_words_ids, index=all_words_set)  # 按字频降序建立所有字的索引，（字-id）
# print("word2id:")
# print(word2id)
id2word = pd.Series(all_words_set, index=all_words_ids)  # (id-字)

# Tag set and ids
tags_set = ["unknown", "other", "Name", "Domain/Topic", "Data-Format", "Language", "Accessibility", "Provenance",
            "Statistics", "Concept", "Geospatial", "Other-Entities", "Temporal", "Other-Numbers", '']
#  # 为解决OOV(Out of Vocabulary)问题，对无效字符标注取unknown
tags_ids = range(len(tags_set))
# print("tag_ids:")
# print(tags_ids)

# Dict to transform
tag2id = pd.Series(tags_ids, index=tags_set)
# print("tag2id:")
# print(tag2id)
id2tag = pd.Series(tags_set, index=tags_ids)
# print("id2tag")
# print(id2tag)

max_length = 20  # 句子最大长度


def x_transform(words):
    # print(words)

    ids = list(word2id[words])
    if len(ids) >= max_length:
        ids = ids[:max_length]
    ids.extend([0] * (max_length - len(ids)))  # 末尾追加00
    return ids


def y_transform(tags):
    # print("__tags:")
    # print(tags)

    ids = list(tag2id[tags])
    if len(ids) >= max_length:
        ids = ids[:max_length]
    ids.extend([0] * (max_length - len(ids)))
    return ids


def transform():
    print('Starting transform...')
    # print(words)
    data_x = list(map(lambda x: x_transform(x), words))  # 字对应的id的序列，words为二维array，多个seq时map并行处理
    # print("data_x:")
    # print(data_x)
    data_y = list(map(lambda y: y_transform(y), labels))  # 字对应的标注的id的序列，二维列表
    # print("data_y:")
    # print(data_y)

    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)

    # print("data_x")
    # print(data_x)
    # print("data_y")
    # print(data_y)


    path = 'data/'
    if not exists(path):
        makedirs(path)

    print('Starting pickle to file...')
    with open(join(path, 'data-small.pkl'), 'wb') as f:
        pickle.dump(data_x, f)  # 序列化对象并追加
        pickle.dump(data_y, f)
        pickle.dump(word2id, f)
        pickle.dump(id2word, f)
        pickle.dump(tag2id, f)
        pickle.dump(id2tag, f)
    print('Pickle finished')








