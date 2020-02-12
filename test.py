import model
import process_data
import numpy as np
from keras.preprocessing.sequence import pad_sequences

#query里面的词组问题！！！！！！！！！！！！！！！！！！！！！！！！！！
#crf！！！！！！！！！！！！！！！！
#停用词！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
maxlen = 20

model1, (vocab, chunk_tags) = model.create_model(train=False)
predict_text = 'Driver Vehicle Licensing Agency Vehicles Database'
word2idx = dict((w, i) for i, w in enumerate(vocab))
print(len(word2idx))

print("predict_txt:"),print(predict_text)
print(predict_text.split(" "))

x = [word2idx.get(w[0].lower(), 1) for w in predict_text.split(" ")]
print("x:"),print(x)

# length = len(x)
# x = pad_sequences([x], maxlen)  # left padding

str, length = process_data.process_data(predict_text.split(" "), vocab)

model1.load_weights('without_crf.h5')
raw = model1.predict(str)[0][-length:]

print("raw:"),print(raw)
result = [np.argmax(row) for row in raw]
result_tags = [chunk_tags[i] for i in result]

# print(result_tags)
for s, t in zip(predict_text.split(" "), result_tags):
    print(s+","+t)


# print(['person:' + per, 'location:' + loc, 'organzation:' + org])
