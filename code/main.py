import spacy
from spacy import displacy
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import nltk
nltk.download('popular', halt_on_error=False)
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from kmeans import Kmeans
from nltk.stem import WordNetLemmatizer
from pca import PCA
from accuracy import confusion_matrix
from graphs import Plot
from frequencymatrix import freq_matrix


word_lemma = WordNetLemmatizer()

porter = PorterStemmer()
lancaster = LancasterStemmer()


class preprocess():
    def __init__(self, text):
        # ner tagging
        NER = spacy.load("en_core_web_sm")

        # defining the lists
        names = []
        desig = []

        for i in text:
            new_text = i.upper()
            ner_tags = NER(new_text)
            #print(ner_tags)
            for word in ner_tags.ents:
                names.append(word.text)
                desig.append(word.label_)

        updated = []
        for i in range(len(text)):
            tokens = word_tokenize(text[i])
            updated.append(tokens)

        i = 0

        thresh = (len(names)-1)
        passing_final = []

    pass


class processing():
    def __init__(self, text):
        # removing stop-words
        self.text = text

    def remove_words(self):
        stop_words = stopwords.words('english')

        # tokenize
        total_output = []
        for i in range(len(self.text)):
            words = word_tokenize(str(self.text[i]))
            output = [w for w in words if not w in stop_words]
            output = " ".join(output)
            total_output.append(output)

        # print(total_output)
        # print(type(total_output))

        return total_output


def n_grams(text, n):
    #print(text)
    output = {}
    final_out = {}
    tokens = []

    for i in range(len(text)):
        tokens.append(word_tokenize(text[i]))

    #print(tokens)

    for i in range(len(text)):
        new_text = word_tokenize(text[i])
        for j in range(len(new_text) - n + 1):
            g = " ".join(new_text[j:j + n])
            output.setdefault(g, 0)
            output[g] += 1
        # print(output)

        final_out.update(output)
    # print(final_out)
    #print(type(final_out))
    new_dict = {}

    for keys, value in final_out.items():
        if value >= 3:
            new_dict[keys] = value

    new_dict = sorted(new_dict.keys(), key=lambda output: output[1], reverse=True)
    #print(new_dict)

    final_pass = []

    for raw in tokens:
            j=0
            propass = []
            i = 0
            while i < (len(raw)):
                word = raw[i]
                if i + 1 < len(raw):
                    word2 = raw[i + 1]
                    check = word + " " + word2
                    #print(check)
                else:
                    check = word
                    #print(check)
                if check in new_dict:
                    propass.append(check)
                    i = i + 2
                else:
                    propass.append(word)
                    i = i + 1

            final_pass.append(propass)



    return new_dict, final_pass


def remove_punctuation(text):
    symbols = "!#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for j in range(len(text)):
        for i in range(len(symbols)):
            text[j] = text[j].replace(symbols[i], "")
            text[j] = text[j].replace('"', "")
        text[j] = text[j].replace(",", "")
    #print(text)
    return text



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # path to all the files
    file_path = open("data.txt", "r")

    combined_list = []
    # reading paths
    for f in file_path:

        # removing space at the end
        path_string = f.rstrip("\n")

        with open(path_string, "r") as sub_files:
            text = ''
            for line in sub_files:
                text = text + line
                text = text.replace("\n", " ")
            combined_list.append(text)

    #print(type(combined_list))

    # print(terms)
    file_path.close()

    processed_list = []
    processed_lemma = []

    for i in range(len(combined_list)):
        data = combined_list[i]
        text = ''
        text1 = ''
        data = data.lower()
        token_words = word_tokenize(data)
        for words in token_words:
            output1 = word_lemma.lemmatize(words)
            output = porter.stem(output1)
            text = text + " " + output1
            text1 = text1 + " " + output

            # output = porter.stem(data)
        processed_list.append(text1)
        processed_lemma.append(text)

    pre = preprocess(processed_lemma)


    no_punc = remove_punctuation(processed_lemma)

    for i in range(len(no_punc)):
        no_punc[i] = no_punc[i].replace("'", "")
    #print(no_punc[13])

    pre_process = processing(no_punc)

    new_out = pre_process.remove_words()

    ngrams, passing = n_grams(new_out, 2)
    print("The most frequent words after n-grams are :", ngrams)

    print("==="*20)

    matrix, tf_idf, sub_data = freq_matrix(passing, combined_list)

    print("==="*20)

    kmean_cluster = Kmeans(k=3,  max_iter=50)

    tf_idf_array = tf_idf.to_numpy()
    fitted = kmean_cluster.fit_kmeans(tf_idf_array)

    predicted = kmean_cluster.predict(tf_idf_array)
    centers = kmean_cluster.centroids
    # print(fitted)
    print(predicted)

    print("==="*20)

    #applying pca
    apply_pca = PCA(tf_idf_array)
    out_pca = apply_pca.calculate()

    #confusion matrix
    confuse = confusion_matrix(predicted)
    print_out = confuse.print()

    print("===" * 20)

    #visualizing
    plotting = Plot(out_pca, centers)

    plotting.graph()





