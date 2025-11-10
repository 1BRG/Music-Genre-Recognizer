import numpy as np


class MultinomialNaiveBayes:
    def __init__(self):
        self.a_priori = {}
        self.vocabulary = set()
        self.conditionals = {}
        self.alpha = 1.0

    def calc_a_priori(self, y_train):
        print("Calculating the a priori probability:")
        genres = {}
        total = len(y_train)
        for genre in y_train:
            if genre not in genres:
                genres[genre] = 1
            else:
                genres[genre] += 1
        for genre in genres:
            self.a_priori[genre] = genres[genre] / total
            print(f"{genre}: {self.a_priori[genre]} ")
        print("---------------------------------------------------------------")

    def calc_cond_voc(self, x_train, y_train):
        print("Calculating the conditioned probability and finding the vocabulary: ")
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
        print("Size of vocabulary: ", len(self.vocabulary))
        print("---------------------------------------------------------------")
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
        print("Training on ", len(x_train), " datas")
        self.calc_a_priori(y_train)
        self.calc_cond_voc(x_train, y_train)
        print("---------------------------------------------------------------")

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
        print(f"Evaluate the model accuracy on {len(x_test)} datas:")
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
        print(f"{acc/total * 100}%")
        print("---------------------------------------------------------------")
        return genres_accuracy, acc / total
