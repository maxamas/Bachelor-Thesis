# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 18:45:14 2017

@author: maxha
"""
"""
search for similar sentences in a dataset, according to cosine similarity of sentence embeddings
"""
import pickle
import numpy as np

def cos_sim(X, Y):  
    similarity = np.dot(X, Y)

    square_mag_x = np.sqrt(np.dot(X, X))
    square_mag_y = np.sqrt(np.dot(Y, Y))

    
    cosine = similarity / (square_mag_x * square_mag_y)
    return cosine

def search_similar(dataset, sentence_nr):

    index = 0
    for row_nr, row in enumerate(dataset):
        for i, sentence in enumerate(row[0]):
            if sentence_nr == index:
                c_s = sentence
                c_v = row[2][i]
            index += 1
                
    similarities_d = {}
    similarities_a = []
    
    for row in dataset:
        for i, vec in enumerate(row[2]):
            sim = cos_sim(c_v, vec)
            similarities_a.append(sim)
            similarities_d[sim] = row[0][i]
            
    similarities_a = np.array(similarities_a)
    similarities_a = np.sort(similarities_a)
    
    print(c_s)
    for i, sim in enumerate(similarities_a[-5:-1]):
        print(sim, ":", similarities_d[sim])
        
if __name__ == "__main__":
        
    data_m = pickle.load(open("data/embeddings/movie_embeddings.p", "rb"))
    data_f = pickle.load(open("data/embeddings/financial_embeddings.p", "rb"))
    
    search_similar(dataset=data_m, sentence_nr=1210)
    search_similar(dataset=data_f, sentence_nr=1210)     
