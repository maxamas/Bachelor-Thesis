# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 15:16:06 2017

@author: maxha
"""
"""
load sentence- and textlevel results, save results as used in thesis to csv
"""

import pickle

Financial_bow_acc = pickle.load(open("results/sentence_level/bow/Financial_bow_acc.p", "rb"))
Financial_d2v_acc = pickle.load(open("results/sentence_level/w2v/Financial_w2v_acc.p", "rb"))
Financial_gicf_acc = pickle.load(open("results/sentence_level/gicf/Financial_gicf_acc.p", "rb"))

Movie_bow_acc = pickle.load(open("results/sentence_level/bow/Movie_bow_acc.p", "rb"))
Movie_d2v_acc = pickle.load(open("results/sentence_level/w2v/Movie_w2v_acc.p", "rb"))
Movie_gicf_acc = pickle.load(open("results/sentence_level/gicf/Movie_gicf_acc.p", "rb"))


accuracies_sentence = [["Model:","Movie Sentence", "Financial Sentence"],
                       ["GICF",Movie_gicf_acc, Financial_gicf_acc],
                       ["Word_2_Vec",Movie_d2v_acc, Financial_d2v_acc],
                       ["BOW", Movie_bow_acc, Financial_bow_acc]]


file = open("results/final/accuracies_sentence_level.csv", "w")
for row in accuracies_sentence:
    line = str(row[0]) + "," + str(row[1]) + "," + str(row[2]) + "\n"
    file.write(line)
file.close()

Financial_bow_acc_t     = pickle.load(open("results/text_level/bow/financial_bow_acc.p", "rb"))
Financial_d2v_acc_t     = pickle.load(open("results/text_level/w2v/financial_w2v_acc.p", "rb"))
Financial_gicf_acc_t    = pickle.load(open("results/text_level/gicf/financial_acc_text.p", "rb")) ###

Movie_bow_acc_t     = pickle.load(open("results/text_level/bow/movie_bow_acc.p", "rb"))
Movie_d2v_acc_t     = pickle.load(open("results/text_level/w2v/movie_w2v_acc.p", "rb"))
Movie_gicf_acc_t    = pickle.load(open("results/text_level/gicf/movie_acc_text.p", "rb")) ###

accuracies_text = [["Model:", "Movie Text", "Financial Text"],
                   ["GICF",Movie_gicf_acc_t, Financial_gicf_acc_t],
                   ["Word_2_Vec",Movie_d2v_acc_t, Financial_d2v_acc_t],
                   ["BOW", Movie_bow_acc_t, Financial_bow_acc_t]]


file = open("results/final/accuracies_text_level.csv", "w")
for row in accuracies_text:
    line = str(row[0]) + "," + str(row[1]) + "," + str(row[2]) + "\n"
    file.write(line)
file.close()
