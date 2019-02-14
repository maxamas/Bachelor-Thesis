# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 20:35:01 2017

@author: maxha
"""
import pickle
import gensim
from gensim.models.word2vec import LineSentence 
import numpy
from sklearn.linear_model import LogisticRegression

def build_word2vec_model(dataset, save_folder):
    """
    build doc2vec model
    """
    
    file = open("results/trained_models/w2v/model_all_review_texts.txt", "w", encoding="utf-8")
    for row in dataset:
        for sentence in row[0]:
            file.write(sentence+"\n")
    file.close()   
    
    doc = open("results/trained_models/w2v/model_all_review_texts.txt", "r", encoding="utf-8")
    sentences = LineSentence(doc)
    
    model = gensim.models.Word2Vec(sentences,
                                  size=500,
                                  window=5,
                                  min_count=0,
                                  workers=4,
                                  iter=10)
    doc.close()
    model.save(fname_or_handle=save_folder)
    return model

def create_text_matrix(data, model): 
    for row in data:
        text_vec = []
        for sentence in row[0]:
            for word in sentence.split():
                if word == "":
                    continue
                else:
                    text_vec.append(model[word])
        text_vec = numpy.mean(numpy.array(text_vec), axis=0)
        row.append(text_vec)
        
        
def create_regression_variables(dataset, data_name):
    arrays = []
    labels = []
    for row in dataset:
        arrays.append(row[2])
        if data_name == "Movie":
            labels.append(1 if float(row[1])>0.5 else 0)
        elif data_name == "Financial":
            labels.append(1 if float(row[1])>0 else 0)
    arrays = numpy.array(arrays)
    labels = numpy.array(labels)
    return arrays, labels
            
def make_model(name="Movie"):
    if name == "Movie":
        train_data = pickle.load( open("data/preprocessed/movie_data.p", "rb"))   
    elif name == "Financial":
        train_data = pickle.load( open("data/preprocessed/financial_data.p", "rb"))
    
    #train the d2v model and make text embeddings as average word embeddings 
    save_name = name + "_model_words"   
    model = build_word2vec_model(dataset=train_data, save_folder="results/trained_models/w2v/" + save_name)
    create_text_matrix(data=train_data, model=model)
    
    #train the logistic regression
    arrays, labels = create_regression_variables(dataset=train_data, data_name=name)
    classifier = LogisticRegression()
    classifier.fit(arrays, labels)
    acc = classifier.score(arrays, labels)
    
    intercept = classifier.intercept_
    coeffiecient = classifier.coef_
    
    return acc, intercept, coeffiecient

  
movie_acc, movie_intercept, movie_coeffiecient = make_model()
financial_acc, financial_intercept, financial_coeffiecient = make_model("Financial")

pickle.dump(movie_intercept, open("results/trained_models/w2v/movie_w2v_intercept.p", "wb"))
pickle.dump(movie_coeffiecient, open("results/trained_models/w2v/movie_w2v_coeffiecient.p", "wb"))
pickle.dump(financial_intercept, open("results/trained_models/w2v/financial_w2v_intercept.p", "wb"))
pickle.dump(financial_coeffiecient, open("results/trained_models/w2v/financial_w2v_coeffiecient.p", "wb"))
    
pickle.dump(movie_acc, open("results/text_level/w2v/movie_w2v_acc.p", "wb"))
pickle.dump(financial_acc, open("results/text_level/w2v/financial_w2v_acc.p", "wb"))

