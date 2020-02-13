import model
import process_data
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# 停用词表
stopwords = []
file = open("stopwords.txt", "r")
for line in file:  # 设置文件对象并读取每一行文件
    line = line.strip('\n')
    stopwords.append(line)

#query里面的词组问题！！！！！！！！！！！！！！！！！！！！！！！！！！
#crf！！！！！！！！！！！！！！！！

#讲道理停用词也该有用啊！！！！！！！！！！！！！不同停用词的词性
maxlen = 20

model1, (vocab, chunk_tags) = model.create_model(train=False)
predict_text = 'large spatio-temporal dataset containing latitude longitude time in csv format'
predictlist1 = predict_text.split(" ")

predictlist = []
#去除停用词
for p in predictlist1:
    predictlist.append(p)

word2idx = dict((w, i) for i, w in enumerate(vocab))
print(word2idx)

print("predict_txt:"),print(predict_text)
print(predictlist)

# x = [word2idx.get(w[0].lower(), 1) for w in predictlist]
# print("x:"),print(x)

# length = len(x)
# x = pad_sequences([x], maxlen)  # left padding

str, length = process_data.process_data(predictlist, vocab)

model1.load_weights('without_crf.h5')
raw = model1.predict(str)[0][-length:]

print("raw:"),print(raw)
result = [np.argmax(row) for row in raw]
result_tags = [chunk_tags[i] for i in result]

# print(result_tags)
for s, t in zip(predictlist, result_tags):
    print("("+s+","+t+")")


