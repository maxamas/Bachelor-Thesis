# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:21:21 2017

@author: maxha
"""
"""
takes aprox. 5min to run, 
"""
import pickle
import numpy
import gensim
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools

def sigmoid_function(x, beta_0, beta_1):
    return (1 / (1 + numpy.exp(-numpy.dot(x, beta_1) + beta_0)))
            
def make_predictions_labels(dataset_name="Movie", model_name="gicf"):
    """
    load sentencelevel dataset Movie/Financial
    load coefficient/intercept of models gicf/w2v/bow obtained from training at textlevel
    create embeddings according to models gicf/w2v/bow for sentencelevel data
    use coefficient/intercept and embeddings to calculate probability of a sentence to belong to class positive (=>scores) the sentiment polarity of each sentence in dataset
    return predicted class (1/0), actual label (1/0), and score(0.0-1.0)
    """
    
    if dataset_name == "Movie":
        dataset = pickle.load(open("data/sentence_level_data/movie/movie_sentence_level_test_data.p", "rb"))
        
        if model_name == "gicf":
            coefficient = pickle.load(open("results/trained_models/gicf/movie_beta_sentence.p", "rb"))
            intercept = 0
            model = gensim.models.Doc2Vec.load("data/doc_2_vec/movie_model")
            embeddings = [model.infer_vector(row[0].split(), alpha=0.1, steps=20) for row in dataset]
            embeddings = numpy.array(embeddings)
            
        elif model_name == "w2v":
            coefficient = numpy.transpose(pickle.load(open("results/trained_models/w2v/movie_w2v_coeffiecient.p", "rb")))
            intercept = pickle.load(open("results/trained_models/w2v/movie_w2v_intercept.p", "rb"))
            model = gensim.models.Word2Vec.load("results/trained_models/w2v/Movie_model_words")
            embeddings = numpy.zeros((500,1), float)
            for row in dataset:
                word_vecs = numpy.zeros((500,1), float)
                for word in row[0].split():
                    try:
                        word_vec = numpy.expand_dims(model.wv[word], axis=1)
                    except:
                        continue
                    word_vecs = numpy.append(word_vecs, word_vec, axis=1)
                    text_vec = numpy.expand_dims(numpy.mean(word_vecs, axis=1), axis=1)
                embeddings = numpy.append(embeddings, text_vec, axis=1)
            embeddings = numpy.transpose(embeddings)
            embeddings = numpy.delete(embeddings, 0, 0)
            
        elif model_name == "bow":
            coefficient = numpy.transpose(pickle.load(open("results/trained_models/bow/movie_bow_coeffiecient.p", "rb")))
            intercept = pickle.load(open("results/trained_models/bow/movie_bow_intercept.p", "rb"))
            model = pickle.load(open("results/trained_models/bow//Movie_bow.p", "rb"))
            embeddings = []
            for row in dataset:
                vec = model.transform([" ".join(row[0])]).toarray()
                vec_av = vec / len(row[0])
                embeddings.append(vec_av)
            embeddings = numpy.reshape(numpy.array(embeddings), newshape=(-1,5000))
            
            
    elif dataset_name == "Financial":
        dataset = pickle.load(open("data/sentence_level_data/financial/financial_sentence_level_val_data.p", "rb"))
        
        if model_name == "gicf":
            coefficient = pickle.load(open("results/trained_models/gicf/financial_beta_sentence.p", "rb"))
            intercept = 0
            model = gensim.models.Doc2Vec.load("data/doc_2_vec/financial_model")
            embeddings = [model.infer_vector(row[0].split(), alpha=0.1, steps=20) for row in dataset]
            embeddings = numpy.array(embeddings)
            
        elif model_name == "w2v":
            coefficient = numpy.transpose(pickle.load(open("results/trained_models/w2v/financial_w2v_coeffiecient.p", "rb")))
            intercept = pickle.load(open("results/trained_models/w2v/financial_w2v_intercept.p", "rb"))
            model = gensim.models.Word2Vec.load("results/trained_models/w2v/Financial_model_words")
            embeddings = numpy.zeros((500,1), float)
            for row in dataset:
                word_vecs = numpy.zeros((500,1), float)
                for word in row[0].split():
                    try:
                        word_vec = numpy.expand_dims(model.wv[word], axis=1)
                    except:
                        continue
                    word_vecs = numpy.append(word_vecs, word_vec, axis=1)
                    text_vec = numpy.expand_dims(numpy.mean(word_vecs, axis=1), axis=1)
                embeddings = numpy.append(embeddings, text_vec, axis=1)
            embeddings = numpy.transpose(embeddings)
            embeddings = numpy.delete(embeddings, 0, 0)
            
        elif model_name == "bow":
            coefficient = numpy.transpose(pickle.load(open("results/trained_models/bow/financial_bow_coeffiecient.p", "rb")))
            intercept = pickle.load(open("results/trained_models/bow/financial_bow_intercept.p", "rb"))
            model = pickle.load(open("results/trained_models/bow/Financial_bow.p", "rb"))
            embeddings = []
            for row in dataset:
                vec = model.transform([" ".join(row[0])]).toarray()
                vec_av = vec / len(row[0])
                embeddings.append(vec_av)
            embeddings = numpy.reshape(numpy.array(embeddings), newshape=(-1,5000))
       
    scores = sigmoid_function(x=embeddings, beta_0=intercept, beta_1=coefficient)
    predictions = numpy.array([1 if score > 0.5 else 0 for score in scores])
    labels = numpy.array([row[1] for row in dataset])

    return predictions, labels, scores

def make_result_sent(predictions, labels, scores):
    """
    use sklearn metrics 
    """
    confusion_matrix = metrics.confusion_matrix(y_true=labels, y_pred=predictions)
    acc = metrics.accuracy_score(y_true=labels, y_pred=predictions)
    falsepositiverate, truepositiverate, _ = metrics.roc_curve(y_true=labels, y_score=scores)
    roc_auc = metrics.auc(falsepositiverate, truepositiverate)
        
    return confusion_matrix, acc, roc_auc

def save_results():
    """
    call the functions above for combinations of datasets and models, save all results
    create plots for the two datasets, showing ROC-curves for each model
    """
    
    dataset_names = ["Movie", "Financial"]
    model_names = ["gicf", "w2v", "bow"]
    
    iterables = [dataset_names, model_names]
    iterables = itertools.product(*iterables)
    combs = [i for i in iterables]
    result_names = ["confusion_matrix", "acc", "auc"]
    plot_data = []
    
    for comb in combs:
    
        print(comb)
        
        pre, lab, sco = make_predictions_labels(dataset_name=comb[0], model_name=comb[1])
        plot_data.append([comb, pre, lab, sco])
        
        results = make_result_sent(predictions=pre, labels=lab, scores=sco)
        
        for result, result_name in zip(results, result_names):
            
            save_name = "results/sentence_level/" + comb[1] + "/" + comb[0] + "_" + comb[1]+ "_" + result_name + ".p"
        
            pickle.dump(result, open(save_name, "wb"))
            
    for name in dataset_names:
        if name == "Movie":
            nb = [0,1,2]
        if name == "Financial":
            nb = [3,4,5]
        _, p_gicf, l_gicf, s_gicf = plot_data[nb[0]]
        fp_gicf, tp_gicf, _ = metrics.roc_curve(y_true=l_gicf, y_score=s_gicf)
            
        _, p_w2v, l_w2v, s_w2v = plot_data[nb[1]]
        fp_w2v, tp_w2v, _ = metrics.roc_curve(y_true=l_w2v, y_score=s_w2v)
            
        _, p_bow, l_bow, s_bow = plot_data[nb[2]]
        fp_bow, tp_bow, _ = metrics.roc_curve(y_true=l_bow, y_score=s_bow)
        
        plt.figure()
        lw = 4
        
        plt.plot(fp_gicf, tp_gicf, color='red', lw=lw, label="gicf")
        plt.plot(fp_w2v, tp_w2v, color='green', lw=lw, linestyle=':', label="w2v")
        plt.plot(fp_bow, tp_bow, color='darkorange', lw=lw, linestyle='-.', label="bow")
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=15)
        plt.ylabel('True Positive Rate', fontsize=15)
    
        plt.legend(loc="lower right", fontsize=15)
        
        #plt.show()
        plot_name = name + "_" + "all_models" + ".eps"
        plt.savefig("results/final/" + plot_name, format='eps', dpi=1000)
   

if __name__ == "__main__":
    save_results()
