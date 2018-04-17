import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import codecs
import re


train_fp = codecs.open("./data/nuclear/train_tmp.txt", "r", "utf-8").readlines()
test_fp = codecs.open("./data/nuclear/test_tmp.txt", "r", "utf-8").readlines()

train_X = []
train_Y = []
for i in range(0, int(len(train_fp) / 3)):
    train_X.append(train_fp[i*3])
    train_Y.append(train_fp[i*3+2])

test_X = []
test_Y = []
for i in range(0, int(len(test_fp) / 3)):
    test_X.append(test_fp[i*3])
    test_Y.append(test_fp[i*3+2])

print(len(train_Y), len(test_Y))
if len(test_X) != len(test_Y) or len(train_X) != len(train_Y):
    print("Sample length error !")
    exit()


def pre_processing(input):
    output = []
    lem_op = WordNetLemmatizer()
    english_stopwords = stopwords.words('english')
    for row in input:
        whole_str = str()
        tmp = re.sub(r"[\/\*\-\+\=\~\`\!\@\#\$\%\^\&\*\(\)\,\.\<\>\/\?\;\:\'\"\s]+", " ", row)
        tmp = (tmp.lower()).split()
        for i in tmp:
            i = lem_op.lemmatize(i)
            if i not in english_stopwords:
                whole_str = whole_str + " " + i
        output.append(whole_str)
    print(output[0:2])
    return output


train_X = pre_processing(train_X)
test_X = pre_processing(test_X)

tf_idf_vectorizer = TfidfVectorizer(norm="l2")
train_X_tfidf = tf_idf_vectorizer.fit_transform(train_X)

svm_op = SVC(kernel="linear")
svm_op.fit(train_X_tfidf, train_Y)

test_X_tfidf = tf_idf_vectorizer.transform(test_X)
predicted = svm_op.predict(test_X_tfidf)
# print(predicted)
# print(test_Y)
print(np.mean(predicted == test_Y))

