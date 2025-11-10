import pandas as pd
import numpy as np
import matplotlib as plt

from pandas.io.formats.format import return_docstring

csv_path = "Music-Databases/Light_Music_Database.csv"



class MultinomialNaiveBayes:
    def __init__(self):
        self.a_priori = {}
        self.vocabulary = set()
        self.conditionals = {}
        self.alpha = 1.0

    def calc_a_priori(self, y_train):
        genres = {}
        total = len(y_train)
        for genre in y_train:
            if (genre not in genres):
                genres[genre] = 1
            else:
                genres[genre] += 1
        for genre in genres:
            self.a_priori[genre] = genres[genre] / total

    def calc_cond_voc(self, x_train, y_train):
        for i in range (0, len(x_train)):
            l = x_train[i]
            genre = y_train[i]
            if(genre not in self.conditionals):
                self.conditionals[genre] = {}
            for word in l:
                self.vocabulary.add(word)
                if word not in self.conditionals[genre]:
                    self.conditionals[genre][word] = 1
                else:
                    self.conditionals[genre][word] += 1

        voc_size = len(self.vocabulary)
        for genre in self.conditionals:
            total_words_genre = sum(self.conditionals[genre]) + self.alpha
            for word in self.conditionals[genre]:
                self.conditionals[genre][word] = self.conditionals[genre][word] / (total_words_genre + voc_size * self.alpha)



    def train(self, x_train, y_train):
        self.calc_a_priori(y_train)
        self.calc_cond_voc(x_train, y_train)

    def predict(self, x_test):
        y_test = []
        for text in x_test:
            vocabulary = {}
            for word in text:
                if word not in vocabulary:
                    vocabulary[word] = 1
                else:
                    vocabulary[word] += 1
            genre_prob = {}
            for genre in self.conditionals:
                prob = self.a_priori[genre]
                for word in vocabulary:
                    if word in self.vocabulary:
                        prob += vocabulary[word] * np.log(self.conditionals[genre][word])
                genre_prob[genre] = prob
            maxx = -1
            final_genre = ""
            for genre in genre_prob:
                if genre_prob > maxx:
                    maxx = genre_prob, final_genre = genre
            y_test.append(final_genre)
        return y_test

    def evaluate(self, x_test, y_test):
        y_result = self.predict(x_test)
        genres_accuracy = {}
        total = len(y_test)
        acc = 0
        for result, i in zip(y_result, (0, len(y_test))):
            correct = y_test[i]
            if correct not in genres_accuracy:
                genres_accuracy[correct] = {}
            if result not in genres_accuracy[correct]:
                genres_accuracy[correct][result] = 1
            else:
                genres_accuracy[correct][result] += 1
            if result is not correct:
                acc += 1
        print("Model Accuracy: ", acc / total * 100, "%")





def afis_details_csv(data):
    genres = {}
    for i, row in data.iterrows():
        genre = row["Genre"]
        if genre in genres:
            genres[genre] += 1
        else:
            genres[genre] = 1
    for genre in genres:
        print(genre, ": ", genres[genre])


def read_csv():
    cols = ["Id", "Genre", "Lyrics"]
    data = pd.read_csv(csv_path, usecols = cols, index_col="Id")
    afis_details_csv(data)
    return data

def tokens_text(text):
    str = text.split()
    return str

def preprocess_data(data):
    data['Tokens'] = data['Lyrics'].apply(tokens_text)

def voc_of_data(data):
    voc = {}
    it = 0

    return voc


data = read_csv()
preprocess_data(data)
#vocabulary = voc_of_data(data)
print(data.items)
