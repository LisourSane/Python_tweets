import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import numpy as np
import scipy
import gensim
from gensim.models import Word2Vec
import pandas as pd
import csv

nltk.download('punkt')

# Open the CSV file
def processing_data(url: str):
    with open(url, 'r', newline='') as csvfile:
        # Specify a custom delimiter
        csv_reader = csv.reader(csvfile, delimiter=',')

        # Initialize a list to store the parsed data
        data = []

        # Iterate over each row in the CSV file
        for row in csv_reader:
            # Initialize a variable to store the current sentence
            current_sentence = ""

            # Iterate over each element in the row
            for element in row:
                # If the element starts with a capital letter, it's a new data field
                if element.strip() and element[0].isupper():
                    # If the current_sentence is not empty, add it to the data list
                    if current_sentence:
                        data.append(current_sentence.strip())

                    # Start a new sentence with the current element
                    current_sentence = element
                else:
                    # Add the element to the current sentence
                    current_sentence += ", " + element

            # Add the last sentence to the data list if it's not empty
            if current_sentence:
                data.append(current_sentence.strip())

    return data
data_pos = processing_data('processedPositive.csv')
data_neu = processing_data('processedNeutral.csv')
data_neg = processing_data('processedNegative.csv')
#tokenizing data
tok_pos = [word_tokenize(i) for i in data_pos]
#stemming data
porter_stemmer = PorterStemmer()
for i in tok_pos:
    for j in i:
        j = porter_stemmer.stem(j)
#print(tok_pos)
#vectorising data
vec_pos = []

# Load pre-trained Word2Vec model
model = Word2Vec.load("path_to_model")
for tweet_tokens in tok_pos:
    tweet_vectors = [model.wv[token] for token in tweet_tokens if token in model.wv.vocab]
    vec_pos.append(tweet_vectors)
print(vec_pos)


