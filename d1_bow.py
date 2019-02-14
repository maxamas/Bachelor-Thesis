# -*- coding: utf-8 -*-
"""
Created on Thu May 25 22:46:07 2017

@author: maxha
"""


from sklearn.feature_extraction.text import CountVectorizer
import pickle
import re
import numpy
import collections
from sklearn.linear_model import LogisticRegression

def bow_model(input_data, save_name):
    corpus = []
    for row in input_data:
        for sentence in row[0]:
            for word in sentence.split():
                if not re.search(r"[0-9]", word):
                    corpus.append(word)
                
    print(collections.Counter(corpus).most_common(5000)[-1])
    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    vectorizer.fit(corpus)
    file_name = "results/trained_models/bow/" + save_name +"_bow.p"
    
    pickle.dump(vectorizer, open(file_name, "wb"))
    return vectorizer

def bow_embeddings(dataset, vectorizer):
    for index,row in enumerate(dataset):
        vec = vectorizer.transform([" ".join(row[0])]).toarray()
        vec_av = vec / len(row[0])
        row.append(vec_av)
        
def create_regression_variables(dataset, data_name):
    arrays = []
    labels = []
    for row in dataset:
        arrays.append(row[2])
        if data_name == "Movie":
            labels.append(1 if float(row[1])>0.5 else 0)
        elif data_name == "Financial":
            labels.append(1 if float(row[1])>0 else 0)
    arrays = numpy.reshape(numpy.array(arrays), newshape=(-1,5000))
    labels = numpy.array(labels)
    return arrays, labels
        
def make_model(name="Movie"):
    if name == "Movie":
        train_data = pickle.load( open("data/preprocessed/movie_data.p", "rb"))   
    elif name == "Financial":
        train_data = pickle.load( open("data/preprocessed/financial_data.p", "rb"))
    
    #train BOW model
    model = bow_model(input_data=train_data, save_name=name)
    bow_embeddings(dataset=train_data, vectorizer=model)
    
    #train the logistic regression
    arrays, labels = create_regression_variables(dataset=train_data, data_name=name)
    classifier = LogisticRegression()
    classifier.fit(arrays, labels)
    acc = classifier.score(arrays, labels)
    
    intercept = classifier.intercept_
    coeffiecient = classifier.coef_
    
    return acc, intercept, coeffiecient
            

m_acc, m_intercept, m_coeffiecient = make_model(name="Movie")
f_acc, f_intercept, f_coeffiecient = make_model(name="Financial")

pickle.dump(m_intercept, open("results/trained_models/bow/movie_bow_intercept.p", "wb"))
pickle.dump(m_coeffiecient, open("results/trained_models/bow/movie_bow_coeffiecient.p", "wb"))
pickle.dump(f_intercept, open("results/trained_models/bow/financial_bow_intercept.p", "wb"))
pickle.dump(f_coeffiecient, open("results/trained_models/bow/financial_bow_coeffiecient.p", "wb"))

pickle.dump(m_acc, open("results/text_level/bow/movie_bow_acc.p", "wb"))
pickle.dump(f_acc, open("results/text_level/bow/financial_bow_acc.p", "wb"))
