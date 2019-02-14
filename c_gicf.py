# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 20:35:01 2017

@author: maxha
"""

from sklearn.cross_validation import train_test_split
import random
import pickle
import numpy
import matplotlib.pyplot
import c2_similarity as similarity
import c1_gradient_helper_functions as ghf
import gensim
import sys
    
    
def split_dataset(dataset, test_size):
    """
    Split dataset into train and validation sets    
    """
    random.shuffle(dataset)
    
    rating_negativ = []
    rating_positiv = []
    
    for row in dataset:
        if int(row[1]) == 0:
             rating_negativ.append(row)
        elif int(row[1]) == 1:
            rating_positiv.append(row)

    random.shuffle(rating_positiv)
    random.shuffle(rating_negativ)               
       
    neg_train_data, neg_val_data   = train_test_split(rating_negativ, test_size=test_size)
    pos_train_data, pos_val_data   = train_test_split(rating_positiv, test_size=test_size)
    
    train_data = neg_train_data + pos_train_data
    val_data = neg_val_data + pos_val_data
    
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    return train_data, val_data 
        

def dataset_as_arrays(dataset):
    """
    Helper function
    Write all text matricies from a dataset into one matrix for the whole dataset.
    (use for gradient descent)
    """
    scores = []
    lenghts = []
    embeddings = []
    for row in dataset:
        embeddings += [vec for vec in row[0]]
        scores.append(float(row[1]))
        lenghts.append(row[0].shape[0])
    
    embeddings = numpy.array(embeddings)
    scores = numpy.array(scores)
    lenghts = numpy.array(lenghts)
    return embeddings, scores, lenghts


def min_batch_gradient_descent_with_momentum(train_data, val_data_text, batch_size, iterations, step_size, lam, momentum_step_size, vec_lenght, d2v_model, val_data_sentence=None):
    """
    Get a beta which minimizes the cost function.
    """
    beta = numpy.zeros(shape=(vec_lenght,1), dtype=float)
    accuracies = []
    betas = []
    losses = []
    sentence_accuracies = []
    
    for iteration in range(iterations):
        batch_start = 0
        batch_end = batch_size
        random.shuffle(train_data)
        
        
        while batch_end != batch_start:
            
            #use helperfunction to create 3 numpy arrays out of the current batch
            X, gs, gl = dataset_as_arrays(train_data[batch_start:batch_end])
                  
            W_ij = similarity.get_sim_over_arry(X, "cosine", None)
            
            # calculate y_hat and derivative
            Y_ij = ghf.sigmoid_over_array(X, beta)
            
            Y_der_ij = ghf.y_derivat(Y_ij, X) #matrix, as result of the derivate of the sigmoid function for all embeddings given the value of the sigmoid function for all embeddings 
            
            # calculate derivative
            similarity_cost = ghf.similarity_derivat(Y_ij, Y_der_ij, W_ij) / (X.shape[0] ** 2) #derivative of the first term 
            group_cost = lam / float(len(gs)) * ghf.group_derivative(Y_ij, Y_der_ij, gs, gl) 
            beta_der = numpy.expand_dims(similarity_cost + group_cost, axis=1)
            
            # new beta
            beta = momentum_step_size * beta - (1 - momentum_step_size) * step_size / (iteration + 1) * beta_der
            betas.append(beta)
            
            #accuracy on text level validation set, given current beta
            accuracies.append(accuracy_text_level(dataset=val_data_text, beta=beta))
            #accuracy on sentence level validation set, given current beta
            if not val_data_sentence==None:
                sentence_accuracies.append(accuracy_sentence_level(dataset=val_data_sentence, beta=beta, d2v_model=d2v_model))
            else:
                sentence_accuracies = None
            
                
            #loss , given current beta
            loss = ghf.loss(similarity=W_ij, sigmoid_values=Y_ij, group_scores=gs, group_lengths=gl, lam=lam)
            losses.append(loss)

            #calculate start and end of the next batch
            if len(train_data)< batch_end + batch_size:
                batch_end = len(train_data)
                batch_start = batch_end
            else:
                batch_end += batch_size
                batch_start += batch_size
                
            #print progress
            sys.stdout.write("\r{0}".format("Iteration: %d/%d |Batch: %03d/%d | Text-level acc: %f | Sentence-level acc: %f | Loss: %f" %((iteration+1), (iterations+1), (batch_start//batch_size), (len(train_data)//batch_size), (accuracies[-1]), (sentence_accuracies[-1]), (loss))))
            sys.stdout.flush()
    
    accuracies = numpy.array(accuracies)
    sentence_accuracies = numpy.array(sentence_accuracies)
    losses = numpy.array(losses)
    return betas, accuracies, sentence_accuracies, losses

def accuracy_sentence_level(dataset, beta, d2v_model):
    """helperfunction calculates sentence level acc for given beta"""
    
    prediction_right = 0
    
    for row in dataset:
        sentence_vec = numpy.reshape(d2v_model.infer_vector(row[0].split(), alpha=0.1, steps=20), newshape=(200,-1))
        prediction = (1 / (1 + numpy.exp(-numpy.dot(numpy.transpose(sentence_vec), beta))))
        prediction = 1 if prediction>0.5 else 0
        if int(row[1]) == prediction:
            prediction_right += 1
                      
    return prediction_right / len(dataset)

def accuracy_text_level(dataset, beta):
    """helperfunction calculates text level acc for given beta"""
    X, gs, gl = dataset_as_arrays(dataset)
    
    #value of the sigmoid function for all embeddings in X, given beta 
    sentence_predictions = ghf.sigmoid_over_array(X, beta)
    
    text_predictions = []
    for index in range(gs.shape[0]):
        frm = numpy.sum(gl[0:index])
        to = frm + gl[index]
        text_prediction = numpy.average(sentence_predictions[frm:to])
        text_predictions.append(0 if text_prediction < 0.5 else 1)
        
    prediction_right = 0
    for index, prediction in enumerate(text_predictions): 
        if prediction == gs[index]:
            prediction_right += 1
            
    return prediction_right/gs.shape[0]     
        


def optimize_dataset(name="Movie"):
    """ finde optimum beta for one of the two datasets"""
    #load textlevel training data, doc2vec model and sentencelevel validation data
    if name=="Movie":
        data = pickle.load(open("data/embeddings/movie_embeddings_small.p", "rb"))
        model = gensim.models.Doc2Vec.load("data/doc_2_vec/movie_model")
        sentence_level_val_data = pickle.load(open("data/sentence_level_data/movie/movie_sentence_level_val_data.p", "rb"))
    elif name=="Financial":
        data = pickle.load(open("data/embeddings/financial_embeddings_small.p", "rb"))
        model = gensim.models.Doc2Vec.load("data/doc_2_vec/financial_model")
        sentence_level_val_data = pickle.load(open("data/sentence_level_data/financial/financial_sentence_level_val_data.p", "rb"))
    
    #split textlevel dataset in training and validation set training for gradien descent optimization validation to select best beta
    train, valid = split_dataset(dataset=data, test_size=0.2)
    betas, text_accuracies, sentence_accuracies, losses = min_batch_gradient_descent_with_momentum(train_data=train,
                                                                   val_data_text=valid,
                                                                   batch_size=5,
                                                                   iterations=2,
                                                                   step_size=0.05, #0.05
                                                                   lam=0.98,
                                                                   momentum_step_size=0.7,
                                                                   vec_lenght=200,
                                                                   d2v_model=model,
                                                                   val_data_sentence=sentence_level_val_data)
    #save textlevel validation data
    pickle.dump(valid, open("data/embeddings/validation_data_" + name + ".p", "wb"))
    
    #select best beta, as beta with highes accuracy in text-level validation data
    for i in range(text_accuracies.shape[0]):
        if text_accuracies[i]  == numpy.max(text_accuracies):
            beta_opt_text_level = betas[i]
            
    #select best beta, as beta with highes accuracy in sentence-level validation data, if sentence-level validation data is given      
    if not sentence_accuracies==None:     
        for i in range(sentence_accuracies.shape[0]):
            if sentence_accuracies[i]  == numpy.max(sentence_accuracies):
                beta_opt_sentecne_level = betas[i]
            
    #plot accuracies for text-level validation data
    matplotlib.pyplot.plot(text_accuracies)
    matplotlib.pyplot.ylabel('text Accuracie')
    matplotlib.pyplot.xlabel('Iterations')
    matplotlib.pyplot.show()
    
    #plot accuracies for sentence-level validation data
    if not sentence_accuracies==None:
        matplotlib.pyplot.plot(sentence_accuracies)
        matplotlib.pyplot.ylabel('Sentence Accuracie')
        matplotlib.pyplot.xlabel('Iterations')
        matplotlib.pyplot.show()
    
    #plot loss (calcualted at text-level)
    matplotlib.pyplot.plot(losses)
    matplotlib.pyplot.ylabel('Cost')
    matplotlib.pyplot.xlabel('Iterations')
    matplotlib.pyplot.show()
        
    acc_max_text = numpy.max(text_accuracies)
    acc_max_sentence = numpy.max(sentence_accuracies)
    
    
    return beta_opt_text_level, beta_opt_sentecne_level, acc_max_text, acc_max_sentence

if __name__ == '__main__':
    
    #calculate results
    movie_beta_text, movie_beta_sentence, movie_acc_text, movie_acc_sentence = optimize_dataset(name="Movie")   
    financial_beta_text, financial_beta_sentence, financial_acc_text, financial_acc_sentence = optimize_dataset(name="Financial")
    
    #save results
    pickle.dump(movie_acc_text, open("results/text_level/gicf/movie_acc_text.p", "wb"))
    pickle.dump(movie_beta_sentence, open("results/trained_models/gicf/movie_beta_sentence.p", "wb"))
    
    pickle.dump(financial_acc_text, open("results/text_level/gicf/financial_acc_text.p", "wb"))
    pickle.dump(financial_beta_sentence, open("results/trained_models/gicf/financial_beta_sentence.p", "wb"))







