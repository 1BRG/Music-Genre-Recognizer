import pandas as pd
import numpy as np

from pandas.io.formats.format import return_docstring

csv_path = "Music-Databases/Light_Music_Database.csv"



class MultinomialNaiveBayes:
    def __init__(self):
        self.a_priori = {}
        self.vocabulary = {}
        self.conditionals = {}

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
            for str in l:



    def train(self, x_train, y_train):
        self.calc_a_priori()
        self.



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
