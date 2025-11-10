import time
from string import punctuation

import pandas as pd




def afis_details_csv(data, details):
    for det in details:
        infos = {}
        print(f"Details about {det}:")
        for i, row in data.iterrows():
            category = row[det]
            if category in infos:
                infos[category] += 1
            else:
                infos[category] = 1
        for category in infos:
            print(category, ": ", infos[category])



def read_csv(csv_path, cols, idx_col = "Id"):
    data = pd.read_csv(csv_path, usecols = cols)
    data = data.dropna()
    return data




import re
import string
stop_words = r"\b(?:i|me|my|myself|we|our|ours|ourselves|you|your|yours|yourself|yourselves|he|him|his|himself|she|her|hers|herself|it|its|itself|they|them|their|theirs|themselves|what|which|who|whom|this|that|these|those|am|is|are|was|were|be|been|being|have|has|had|having|do|does|did|doing|a|an|the|and|but|if|or|because|as|until|while|of|at|by|for|with|about|against|between|into|through|during|before|after|above|below|to|from|up|down|in|out|on|off|over|under|again|further|then|once|here|there|when|where|why|how|all|any|both|each|few|more|most|other|some|such|no|nor|not|only|own|same|so|than|too|very|s|t|can|will|just|don|should|now)\b"
def tokens_text(text, afis = False):
    text = re.sub(r'\[.*?\]', '', text)
    text = text.lower()
    text = re.sub(stop_words, '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    str_list = text.split()
    if afis:
        print(text)
    return str_list

def preprocess_data(data, column = "Lyrics"):
    print("Preprocessing the data...")
    data['Tokens'] = data[column].apply(tokens_text)
    print("---------------------------------------------------------------")



