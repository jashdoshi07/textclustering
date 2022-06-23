from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from math import log
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

"""
define the workflows and utility functions
"""

def freq_matrix(data, data_sub):
    word_set = {}

    for i in range(len(data)):
        word_set = set(data[i]).union(set(word_set))

    word_array = []
    for i in range(len(data)):
        word_array.append(dict.fromkeys(word_set, 0))

    for i in range(len(data)):
        for j in data[i]:
            word_array[i][j] += 1

    top_5 = []

    for i in word_array:
        top_5_dict = {}

        top_5_dict = sorted(i.items(), key=lambda output: output[1], reverse=True)
        top_5.append(top_5_dict)

    final_5 = []
    for i in top_5:
        inbetween = []
        varia = 0
        for word in i:
            inbetween.append(word[0])
            varia = varia +1
            if varia > 5:
                break

        final_5.append(inbetween)

    text_dataframe = pd.DataFrame(final_5)

    file = open("frequntwords.txt", "w")
    file.close()

    text_dataframe.to_csv('frequntwords.txt', sep=' ')

    topic1 = final_5[:8]
    topic2 = final_5[8:16]
    topic3 = final_5[16:]


    counter1 = Counter(topic1[0])
    for i in topic1[1:]:
        counter1.update(i)

    title1 = counter1.most_common()
    title1 = title1[0][0]

    print("Topic for first folder is", title1)

    counter2 = Counter(topic2[0])
    for i in topic2[1:]:
        counter2.update(i)

    title2 = counter2.most_common()
    title2 = title2[0][0]

    print("Topic for second folder is", title2)

    counter3 = Counter(topic3[0])
    for i in topic3[1:]:
        counter3.update(i)

    title3 = counter3.most_common()
    title3 = title3[0][0]

    print("Topic for third folder is ", title3)

    dataframe = pd.DataFrame(word_array)


    maximum_str = dataframe.idxmax(axis=1)

    corpus_total_docs = 0
    tf_total_list = []

    idf = dict.fromkeys(word_set, 0)

    # counting tf
    for i in range(len(data)):

        total_words = len(data[i])
        corpus_total_docs = corpus_total_docs + 1
        tf_list = {}

        for words, counts in word_array[i].items():
            tf_list[words] = counts / total_words

            if counts > 0:
                idf[words] += 1

        tf_total_list.append(tf_list)

    tf_dataframe = pd.DataFrame(tf_total_list)

    for words, counts in idf.items():
        idf[words] = log(corpus_total_docs / (float(counts)))

    
    # calculating tf-idf
    tf_idf = []

    for i in range(len(data)):
        tfidf_dict = dict.fromkeys(tf_total_list[i].keys(), 0)

        for words, counts in tf_total_list[i].items():
            tfidf_dict[words] = counts * idf[words]

        tf_idf.append(tfidf_dict)

    # print(tf_idf)

    tfidf_dataframe = pd.DataFrame(tf_idf)
    # print(tfidf_dataframe)


    tf_idf_sub = TfidfVectorizer()
    process_sub = tf_idf_sub.fit_transform(data_sub)

    process_sub = process_sub.toarray()

    return dataframe, tfidf_dataframe, process_sub
