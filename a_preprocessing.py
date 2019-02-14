# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 20:35:01 2017

@author: maxha
"""
"""
take input datasets from raw folder and performe preprocessing save processed files in preprocessed folder
"""

import re
import csv
import pickle
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import wordpunct_tokenize
from sklearn.cross_validation import train_test_split
import random
import numpy

def clean_word(word):
    word = word.lower()
    
    """
    if re.search("[\d]", word):
        word =  "NUMBER"
    """
    
    word = re.sub(r'(\S)\1+', r'\1\1', word)  # normalize repeated characters to two
    word = re.sub(r'(\S\S)\1+', r'\1\1', word) # normalize repeated double characters to two double characters

    word = word.replace("n't", " not")  # normalize 
    word = word.replace("'", "")
    word = word.replace('\n', '')
    word = word.replace('\k', '')
    word = word.replace('"', '')
    word = word.replace('(', '')
    word = word.replace(')', '')
    word = word.replace('[', '')
    word = word.replace(']', '')
    word = word.replace("'", "")
    
    if not re.search(r"[A-Za-z0-9]", word):
        word = ""

    return word

def split_sentences_to_strings(input_file, name="Movie"):
    """
    Transforme a plane text formation into a list of strings, each string a sentence.
    """
    data = []
    text = str()
    with open(input_file, newline='', encoding="utf-8") as in_file:
        reader = csv.reader(in_file, delimiter=',', quotechar='"')
        next(reader) #skip header
        for row in reader:
            if name=="Movie":
                text += row[2]
            elif name=="Financial":
                text += row[3]
    sent_detector = PunktSentenceTokenizer(train_text=text)
    with open(input_file, newline='', encoding="utf-8") as in_file:
        reader = csv.reader(in_file, delimiter=',', quotechar='"')
        next(reader) #skip header
        for row in reader:
            cleaned_sentences = []
            if name=="Movie":
                sentences = sent_detector.tokenize(row[2].strip())
            elif name=="Financial":
                sentences = sent_detector.tokenize(row[3].strip())
            for sentence in sentences:
                words = wordpunct_tokenize(sentence)
                words_out = []
                for word in words:
                    words_out.append(clean_word(word))
                cleaned_sentence = " ".join(words_out)
                cleaned_sentence = wordpunct_tokenize(cleaned_sentence) #remove double space
                cleaned_sentence = " ".join(cleaned_sentence)
                
                cleaned_sentence = wordpunct_tokenize(cleaned_sentence) #remove double NUMBER token
                last_len = 0
                while last_len != len(cleaned_sentence):
                    last_len = len(cleaned_sentence)
                    for index,word in enumerate(cleaned_sentence):
                        if word == "NUMBER":
                            if index+1 < len(cleaned_sentence):
                                if cleaned_sentence[index+1]=="NUMBER":
                                    del(cleaned_sentence[index+1])
                                    
                cleaned_sentence = " ".join(cleaned_sentence)                 
                cleaned_sentences.append(cleaned_sentence)
            
            if name=="Movie":
                data.append([cleaned_sentences, row[1]])
            elif name=="Financial":
                data.append([cleaned_sentences, row[4], row[3]])    
    return data

def unselect_05_ratings_and_balance_dataset(dataset):
    pos = []
    neg = []
    for row in dataset:
        if  float(row[1]) > 0.5:
            pos.append(row)
        elif float(row[1]) < 0.5:
            neg.append(row)
            
    if len(neg)>len(pos):
        neg = neg[:len(pos)]
    else:
        pos = pos[:len(neg)]
                
    out = neg + pos
    random.shuffle(out)
    
    return out

def select_top_3000_abnormalreturns_and_balance_dataset(dataset):
    returns = numpy.sort(numpy.array([float(r[1]) for r in dataset]))
    lower_bound = returns[2999]
    upper_bound = returns[-2999]
    
    neg = []
    pos = []
    
    for row in dataset:
        if float(row[1])<=lower_bound:
            neg.append(row)
        elif float(row[1])>=upper_bound:
            pos.append(row)
        else:
            continue
        
    neg_test_data, neg_val_data   = train_test_split(neg, test_size=0.2)
    pos_test_data, pos_val_data   = train_test_split(pos, test_size=0.2)

    test_data = neg_test_data + pos_test_data
    val_data = neg_val_data + pos_val_data       
    
    random.shuffle(test_data)
    random.shuffle(val_data)
    
    return test_data, val_data

def clean_dataset(name="Movie"):
    if name == "Movie":
        a = split_sentences_to_strings(input_file="data/raw/movie.csv", name=name)
        a = unselect_05_ratings_and_balance_dataset(dataset=a)
        return a
        
    elif name == "Financial":
        a = split_sentences_to_strings(input_file="data/raw/financial.csv", name=name)
        train, test = select_top_3000_abnormalreturns_and_balance_dataset(dataset=a)
        return train, test
        
    
if __name__ == '__main__':
    """
    clean up the plane text inputs for furthure processing
    required output: each text is a list of strings => required input style for doc2vec
    """
    movie = clean_dataset("Movie")
    pickle.dump(movie, open("data/preprocessed/movie_data.p", "wb"))
    financial_train, financial_test = clean_dataset("Financial")
    pickle.dump(financial_train, open("data/preprocessed/financial_data.p", "wb"))

