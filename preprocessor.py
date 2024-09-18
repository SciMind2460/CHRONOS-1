import tensorflow
import numpy
from tensorflow.python.keras.layers import Sequential
from tensorflow.python.keras.layers import SimpleRNN, Dense
import re 
import string 
import inflect 

p = inflect.engine()
def convert_number_to_text(text):
    temp_str = text.split()
    new_string = []
    for word in temp_str:
        if word.isdigit():
            temp = p.number_to_words(word)
            new_string.append(temp)
        else:
            new_string.append(word)
    temp_str = ' '.join(new_string)
    return temp_str

def preprocessed_text(dataset):
    dataset = dataset.lower()
    dataset = convert_number_to_text(dataset)
    dataset = dataset.translate(str.maketrans('', '', string.punctuation))
    list_of_words = dataset.split(' ')
    return list_of_words

if __name__ = "__main__":
    preprocessor_sample = "2 people walked into a bar. Manav says hello to me."
    process_sample = preprocessed_text(preprocessor_sample)
    print(process_sample)