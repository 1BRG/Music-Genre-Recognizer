

# Project Documentation: Multinomial Naive Bayes Classifier for Music Genre Prediction

**Author:** Bălăceanu Rafael Gabriel

---

# Mathematical Model: Multinomial Naive Bayes

For classifying music genres, the formula is:

$$
P(\text{Genre} | \text{Lyrics}) = \frac{P(\text{Lyrics} | \text{Genre}) \times P(\text{Genre})}{P(\text{Lyrics})}
$$

Where:
- **P(Genre | Lyrics)**: the probability that a song belongs to a certain *Genre*, given its *Lyrics*. This is the result we want to calculate.
- **P(Lyrics | Genre)**: the probability of encountering the given *Lyrics* in a song belonging to a certain *Genre*.
- **P(Genre)** is the prior probability: the overall probability that a song belongs to a certain *Genre* within our dataset.
- **P(Lyrics)** is the probability of the lyrics occurring. This is constant, so we can ignore/eliminate it.

## The "Naive" Assumption

The model makes a "naive" assumption of conditional independence: it considers that the presence of each word in the lyrics is independent of the presence of the other words. Thus, the probability that a song belongs to a certain music genre is based on the probability of each word belonging to it.

$$
P(\text{Lyrics} | \text{Genre}) = P(\text{Genre})\prod_{i=1}^{n} P(\text{word}_i | \text{Genre})
$$

## Logarithmic Probabilities

Multiplying many small probabilities (between 0 and 1) can lead to calculation errors (numeric underflow). To avoid this, the implementation calculates the sum of the logarithms of the probabilities:

$$
\log(P(\text{Lyrics} | \text{Genre})) = \log(P(\text{Genre})\prod_{i=1}^{n} P(\text{word}_i | \text{Genre}))
$$

$$
\log(P(\text{Genre} | \text{Lyrics})) = \log(P(\text{Genre})) + \sum_{i=1}^{n} \log(P(\text{word}_i | \text{Genre}))
$$

The genre with the highest logarithmic probability is chosen as the final prediction.

## Laplace (Additive) Smoothing

To handle words that appear in the test set but are restricted to certain genres (which would result in a zero probability for the other genres), Laplace smoothing is applied. A small value, **alpha** (set to 1.0 in the code), is added to the numerator of each word, preventing zero probabilities.

# Code Structure and Main Functions

The project is organized into three main Python files:

## `data_processing.py`
This file contains all the functions related to loading, cleaning, and preprocessing the data.

- **`read_csv(csv_path, cols, ...)`**: Reads the specified columns (`"Genre"`, `"Lyrics"`) from the CSV file into a pandas DataFrame and drops rows with missing data.
- **`tokens_text(text, ...)`**: The core text processing function. It takes the raw text (lyrics) and performs the following operations:
  1. Removes special characters and content inside square brackets (e.g., "`[Chorus]`"), which are specific to lyrics from the Genius website.
  2. Converts the text to lowercase.
  3. Removes punctuation.
  4. Removes common English words ("stop words").
  5. Splits the cleaned text into a list of words (tokens).
- **`preprocess_data(data, column, ...)`**: Applies the `tokens_text` function to the lyrics column of the DataFrame and stores the result in a new column named `"Tokens"`.

## `MultinomialNBayes.py`
This module contains the implementation of the classifier.
- **`MultinomialNaiveBayes` class**:
  - **`__init__(self)`**: Initializes the model parameters: the `prior` probabilities (a_priori), the `vocabulary`, the `conditional` probabilities, and the smoothing factor `alpha`.
  - **`train(self, x_train, y_train)`**: Orchestrates the training process by calling the following two methods:
    - **`calc_a_priori(self, y_train)`**: Calculates the prior probability $P(\text{Genre})$ for each genre.
    - **`calc_cond_voc(self, x_train, y_train)`**: Builds the vocabulary and calculates the conditional probability $P(\text{word} | \text{Genre})$.
  - **`predict(self, x_test)`**: Takes a list of new lyrics and predicts the genre for each one.
  - **`evaluate(self, x_test, y_test)`**: Measures the model's performance by comparing its predictions with the actual labels.

## `main.py`
This is the main script that ties all components together.
- **Data Loading and Preprocessing**: Reads the `Light_Music_Dataset.csv` / `Heavy_Music_Dataset1.csv` file and preprocesses the lyrics.
- **Model Training and Evaluation**:
  - **Option 1** (default): Splits the dataset into a training set (80%) and a test set (20%) using `train_test_split`.
  - **Option 2**: Splits the dataset into a training set where each genre has a specific proportion from the original set, and a test set containing the remaining data, using `create_false_imbalance(lyrics, genres, procent_per_genre)`.
  - Initializes, trains, and evaluates the `MultinomialNaiveBayes` model.
- **`plot_confusion_from_dict_proportions(...)`**: Visualizes the evaluation results as a confusion matrix. (Code generated with ChatGPT)
- **`start_testing()`**: Starts an interactive loop where the user can input lyrics to get a prediction.

# Usage Instructions
1. **Prerequisites**: Ensure you have Python installed.
2. **Dataset**: The file used, `Light_Music_Dataset.csv` or `Heavy_Music_Dataset1.csv`, must be located in a folder named `Music-Datasets` (sources for downloading the data files can be found at the beginning of the `main.py` file).
3. **Install Dependencies**: Open a terminal and run the following command:
    ```bash
    pip install -r requirements.txt
    ```
4. **Run the Project**: Execute the main script from the terminal:
    ```bash
    python main.py
    ```
The script will first train and evaluate the model, displaying the results. Afterward, it will wait for the user to input text for classification.

# Usage Example
After running the `python main.py` command, the training and evaluation process will be displayed in the terminal. Finally, the interactive prompt will start.

## Terminal Output
```shell
Details about Genre:
Metal :  100000
rock :  99997
rap :  99975
pop :  100000
country :  100000
Preprocessing the data...
---------------------------------------------------------------
Training on  399977  datas
Calculating the a priori probability:
country: 0.19959397665365758 
Metal: 0.19994399677981484 
rock: 0.20009900569282735 
pop: 0.20018651072436666 
rap: 0.20017651014933358 
---------------------------------------------------------------
Calculating the conditioned probability and finding the vocabulary: 
Size of vocabulary:  432670
---------------------------------------------------------------
---------------------------------------------------------------
Evaluate the model accuracy on 99995 datas:
60.44102205110256%
---------------------------------------------------------------
```
<img width="1536" height="754" alt="Heavy_Dataset_no_imbalance" src="https://github.com/user-attachments/assets/0cf5fd97-4de6-45c9-a767-916d736a3457" />

*Confusion Matrix Plot*

## Interactive Session
The program will now wait for your input on a single line (you can use the `oneLineLyrics.cpp` program). To exit, type `EOF` and press Enter.
```shell
> Darkness, imprisoning me All that I see, absolute horror
metal

> Country roads, take me home To the place I belong
country

> EOF
```

# References
- https://www.geeksforgeeks.org/machine-learning/naive-bayes-scratch-implementation-using-python/
- LLMs like ChatGPT and Gemini
