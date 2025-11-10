import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
            if genre not in genres:
                genres[genre] = 1
            else:
                genres[genre] += 1
        for genre in genres:
            self.a_priori[genre] = genres[genre] / total

    def calc_cond_voc(self, x_train, y_train):
        for i in range (0, len(x_train)):
            l = x_train[i]
            genre = y_train[i]
            if genre not in self.conditionals:
                self.conditionals[genre] = {}
            for word in l:
                self.vocabulary.add(word)
                if word not in self.conditionals[genre]:
                    self.conditionals[genre][word] = 1
                else:
                    self.conditionals[genre][word] += 1

        voc_size = len(self.vocabulary)
        for genre in self.conditionals:

            total_words_genre = 0

            for word in self.conditionals[genre]:
                total_words_genre += self.conditionals[genre][word]

            for word in self.vocabulary:
                count_genre = self.alpha
                if word in self.conditionals[genre]:
                    count_genre += self.conditionals[genre][word]
                self.conditionals[genre][word] = count_genre / (total_words_genre + voc_size * self.alpha)



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
                prob = np.log(self.a_priori[genre])
                for word in vocabulary:
                    if word in self.vocabulary:
                        prob += vocabulary[word] * np.log(self.conditionals[genre][word])
                genre_prob[genre] = prob
            maxx = -np.inf
            final_genre = ""
            for genre in genre_prob:
                if genre_prob[genre] > maxx:
                    maxx = genre_prob[genre]
                    final_genre = genre
            y_test.append(final_genre)
        return y_test

    def evaluate(self, x_test, y_test):
        y_result = self.predict(x_test)
        genres_accuracy = {}
        total = len(y_test)
        acc = 0
        for i, result in enumerate(y_result):
            correct = y_test[i]
            if correct not in genres_accuracy:
                genres_accuracy[correct] = {}
            if result not in genres_accuracy[correct]:
                genres_accuracy[correct][result] = 1
            else:
                genres_accuracy[correct][result] += 1
            if result == correct:
                acc += 1
        return (genres_accuracy, acc / total)





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



def plot_confusion_from_dict(genres_accuracy, title='Confusion matrix (counts)', cmap='viridis', figsize=(12,8)):
    # 1) construim lista completă de etichete (true + predicted)
    labels = sorted(set(list(genres_accuracy.keys()) +
                        [p for preds in genres_accuracy.values() for p in preds.keys()]))
    idx = {lab: i for i, lab in enumerate(labels)}

    # 2) construim matricea (n_true x n_pred)
    n = len(labels)
    M = np.zeros((n, n), dtype=int)
    for true_label, preds in genres_accuracy.items():
        for pred_label, count in preds.items():
            M[idx[true_label], idx[pred_label]] = count

    # 3) desenăm cu matplotlib
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, interpolation='nearest', cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Count', rotation=270, labelpad=15)

    # 4) etichete pe axe
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted genre')
    ax.set_ylabel('True genre')
    ax.set_title(title)

    # 5) annotări (numere în celule)
    maxval = M.max() if M.size else 1
    for i in range(n):
        for j in range(n):
            color = 'white' if M[i, j] > maxval / 2 else 'black'
            ax.text(j, i, str(M[i, j]), ha='center', va='center', color=color, fontsize=9)

    plt.tight_layout()
    plt.show()





data = read_csv()
preprocess_data(data)

lyrics, genres = data['Tokens'].tolist(), data['Genre'].tolist()
X_train, X_test, y_train, y_test = train_test_split(lyrics, genres, test_size=0.2, random_state=42)

model = MultinomialNaiveBayes()
model.train(X_train, y_train)
print("")
genres_accuracy, accuracy = model.evaluate(X_test, y_test)
plot_confusion_from_dict(genres_accuracy)

print("Model Accuracy: ", accuracy * 100, "%")
