import numpy as np
import matplotlib.pyplot as plt
import MultinomialNBayes as mnb
import data_processing as dt
from sklearn.model_selection import train_test_split


csv_path = "Music-Datasets/Light_Music_Dataset.csv"
dataset = "Light"
#csv_path = "Music-Datasets/Light_Music_Dataset.csv"
#csv_path = "Heavy"

def percents_of_each_genre(dt_set):
    if dt_set == "Heavy":
        aux_genres_max = {
                "Metal": 0.47,
                "rock": 0.81,
                "rap": 0.92,
                "country": 0.64,
                "pop": 0.78
            }
        return aux_genres_max
    else:
        aux_genres_max = {
                "blues": 0.47,
                "country": 0.81,
                "jazz": 0.92,
                "pop": 0.64,
                "reggae": 0.78,
                "rock": 0.67,
                "hip hop": 0.97
            }
        return aux_genres_max

def plot_confusion_from_dict_proportions(genres_accuracy,
                                         title='Confusion matrix (proportions)',
                                         cmap='Greens',
                                         figsize=(12,8),
                                         normalize='true'):  # normalize: 'true' (pe rând), 'all' (pe total), 'pred' (pe col)
    # construim lista completă de etichete (true + predicted)
    labels = sorted(set(list(genres_accuracy.keys()) +
                        [p for preds in genres_accuracy.values() for p in preds.keys()]))
    idx = {lab: i for i, lab in enumerate(labels)}

    # construim matricea (n_true x n_pred)
    n = len(labels)
    M = np.zeros((n, n), dtype=float)
    for true_label, preds in genres_accuracy.items():
        for pred_label, count in preds.items():
            M[idx[true_label], idx[pred_label]] = count

    # normalizare
    if normalize == 'true':
        # pentru fiecare rând (true class) transformăm în proporții (suma rând = 1)
        row_sums = M.sum(axis=1, keepdims=True)
        # evităm împărțirea la 0
        row_sums[row_sums == 0] = 1.0
        M_prop = M / row_sums
        colorbar_label = 'Proportion (per true class)'
    elif normalize == 'pred':
        # normalizare pe coloane (proporție din predicțiile pentru fiecare coloană)
        col_sums = M.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        M_prop = M / col_sums
        colorbar_label = 'Proportion (per predicted class)'
    elif normalize == 'all':
        total = M.sum()
        total = total if total != 0 else 1.0
        M_prop = M / total
        colorbar_label = 'Proportion (of all samples)'
    else:
        raise ValueError("normalize must be one of: 'true', 'pred', 'all'")

    # desenăm cu matplotlib (folosim M_prop, valori între 0 și 1)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M_prop, interpolation='nearest', cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(colorbar_label, rotation=270, labelpad=15)

    # etichete pe axe
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted genre')
    ax.set_ylabel('True genre')
    ax.set_title(title)

    # annotări (procentaj în fiecare celulă)
    maxval = M_prop.max() if M_prop.size else 1.0
    for i in range(n):
        for j in range(n):
            pct = 100.0 * M_prop[i, j]
            # culori albe pentru valori mari (mai vizibile pe fundal închis)
            color = 'white' if M_prop[i, j] > maxval / 2 else 'black'
            ax.text(j, i, f"{pct:.1f}%", ha='center', va='center', color=color, fontsize=9)

    plt.tight_layout()
    plt.show()

def create_false_imbalance(lyrics, genres, procent_per_genre):
    max_genre = dt.afis_details_csv(data, ["Genre"], False)
    for genre in max_genre:
        max_genre[genre] *= procent_per_genre[genre]
    x_train, x_test, y_train, y_test = [], [], [], []
    for tokens, tip in zip(lyrics, genres):
        if max_genre[tip] > 0:
            x_train.append(tokens)
            y_train.append(tip)
            max_genre[tip] -= 1
        else:
            x_test.append(tokens)
            y_test.append(tip)
    return x_train, x_test, y_train, y_test

def train_model(t_model, imbalance, t_genres_max):
    lyrics, genres = data['Tokens'].tolist(), data['Genre'].tolist()
    if imbalance:
        x_train, x_test, y_train, y_test = create_false_imbalance(lyrics, genres, t_genres_max)
    else:
        x_train, x_test, y_train, y_test = train_test_split(lyrics, genres, test_size=0.2, random_state=42)
    t_model.train(x_train, y_train)

    genres_accuracy, accuracy = model.evaluate(x_test, y_test)
    plot_confusion_from_dict_proportions(genres_accuracy)


def start_testing():
    text = ""
    while text != 'EOF':
        text = input()
        lyrics = dt.tokens_text(text)
        genre = model.predict([lyrics])
        for res in genre:
            print(res)




data = dt.read_csv(csv_path, ["Genre", "Lyrics"])
dt.afis_details_csv(data, ["Genre"])
dt.preprocess_data(data, "Lyrics")
#dt.output_to_file(data)
genres_max = percents_of_each_genre(dataset)


model = mnb.MultinomialNaiveBayes()
train_model(model, False, genres_max)

start_testing()
