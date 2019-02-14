# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:40:28 2017

@author: maxha
"""
import pickle
import numpy


def sigmoid_function(x, beta_0, beta_1):
    return (1 / (1 + numpy.exp(-numpy.dot(x, beta_1) + beta_0)))

def pred_text(dataset, coef, text_nb):
    
    for i, vec in enumerate(dataset[text_nb][2]):
        pred = sigmoid_function(vec, 0, coef)
        if pred > 0.5:
            print(dataset[text_nb][0][i].upper())
        else: print(dataset[text_nb][0][i])
        
def short_texts(dataset, text_lenght):
    short_texts = []
    for row in dataset:
        if len(row[0]) < text_lenght:
            short_texts.append(row)
    return short_texts



data_m = pickle.load(open("data/embeddings/movie_embeddings.p", "rb"))
data_f = pickle.load(open("data/embeddings/financial_embeddings.p", "rb"))

coefficient_m = pickle.load(open("results/trained_models/gicf/movie_beta_sentence.p", "rb"))
coefficient_f = pickle.load(open("results/trained_models/gicf/financial_beta_sentence.p", "rb"))

f_short = short_texts(dataset=data_f, text_lenght=5)


pred = sigmoid_function(data_f[0][2][0], 0, coefficient_f)
numpy.isclose(0.5, pred, atol=0.01)

# reslut chapter
pred_text(data_f, coefficient_f, text_nb=9)

pred_text(f_short, coefficient_f, text_nb=17)
pred_text(f_short, coefficient_f, text_nb=2)
pred_text(f_short, coefficient_f, text_nb=48)
pred_text(f_short, coefficient_f, text_nb=52)


