# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 20:35:01 2017

@author: maxha
"""

"""
take input datasets from preprocessed folder, create a doc2vec model, get vector representations of senteces from doc2vec model,
arange vector representations as matricies => each matrix is one text, save output in embeddings folder,
small dataset (input for gicf) => [[textmatrix.1, sentimentpolarity text.1],[textmatrix.2, sentimentpolarity text.2], ....] 
other dataset (not further used, just as backup) => contains also original text and original rating/abnormal return 
"""

import pickle
import gensim
from gensim.models.doc2vec import TaggedLineDocument 
import numpy
assert gensim.models.doc2vec.FAST_VERSION > -1


def make_doc2vec_inputfile(dataset, save_file):
    """
    create one text file for all training documents (input for gensim doc2vec), 
    """    
    file_name = save_file
    file = open(file_name, "w+", encoding="utf-8")
    for row in dataset:
        for sentence in row[0]:
            line = sentence + "\n"
            file.write(line)
    
    file.close()
    return file_name

def build_doc2vec_model(dataset, vec_lenght, save_folder, name="Movie"):
    """
    build doc2vec model
    """    
    #use helperfunction write_all_in_txt, which creates an txt file with required format for doc2vec model
    if name == "Movie":
        txt_file = make_doc2vec_inputfile(dataset=dataset, save_file="data/doc_2_vec/movie_d2v_input.txt")
    elif name == "Financial":
        txt_file = make_doc2vec_inputfile(dataset=dataset, save_file="data/doc_2_vec/financial_d2v_input.txt")
    
    doc = open(txt_file, "r", encoding="utf-8")
    documents = TaggedLineDocument(doc)
    
    model = gensim.models.Doc2Vec(documents,
                                  dm=0,
                                  dbow_words=0,
                                  size=vec_lenght,
                                  window=10,
                                  hs=0,
                                  negative=5,
                                  sample=1e-4,
                                  iter=20,
                                  min_count=10,
                                  workers=4,
                                  alpha=0.1)
    doc.close()
    model.save(fname_or_handle=save_folder)
    return model

def create_sentence_matrix(dataset, model, vec_lenght):
    """
    Create a numpy matrix for each review text, write to train data.
    """   
    doc_nr = 0
    for row in dataset:
        text_matrix = []
        for sentence in row[0]:
            text_matrix.append(model.docvecs[doc_nr])
            doc_nr += 1
        row.append(numpy.array(text_matrix))
   
def check_sentence_matrix(dataset, model):
    """
    Check if first column off the text matrix is equal to the 
    first sentence of the doc2vec model.
    
    """
    print(dataset[0][-1][0] == model.docvecs[0])
    
def make_classification(dataset, name="Movie"):
    if name == "Movie":
        split_value = 0.5
    elif name == "Financial":
        split_value = 0
        
    for row in dataset:
        if float(row[1])>split_value:
            row.append(1)
        else: row.append(0)
        
def make_small_dataset(dataset):
    return [[row[-2], row[-1]] for row in dataset]
    

def make_embeddings(name="Movie"):
    if name=="Movie":
        a = pickle.load(open("data/preprocessed/movie_data.p", "rb"))
        model = build_doc2vec_model(dataset=a, vec_lenght=200, save_folder="data/doc_2_vec/movie_model", name=name)
        create_sentence_matrix(dataset=a, model=model, vec_lenght=200)
        check_sentence_matrix(dataset=a, model=model)
        make_classification(a, name=name)
        pickle.dump(a, open("data/embeddings/movie_embeddings.p", "wb"))
        a_small = make_small_dataset(dataset=a)
        pickle.dump(a_small, open("data/embeddings/movie_embeddings_small.p", "wb"))
    if name=="Financial":
        a = pickle.load( open("data/preprocessed/financial_data.p", "rb"))
        model = build_doc2vec_model(dataset=a, vec_lenght=200, save_folder="data/doc_2_vec/financial_model", name=name)
        create_sentence_matrix(dataset=a, model=model, vec_lenght=200)
        check_sentence_matrix(dataset=a, model=model)
        make_classification(a, name=name)
        pickle.dump(a, open("data/embeddings/financial_embeddings.p", "wb"))
        a_small = make_small_dataset(dataset=a)
        pickle.dump(a_small, open("data/embeddings/financial_embeddings_small.p", "wb"))
##########################################################################################
if __name__ == "__main__":
    make_embeddings(name="Movie")
    make_embeddings(name="Financial")
