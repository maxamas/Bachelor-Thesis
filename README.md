# Bachelor-Thesis


## a_preprocessing.py : 
loads raw data from data/raw and performes preprocessing(remove punctuation etc.), determines sentiment polarity of texts(using ratings for movies and abnormal return for finacial news), save processed files in preprocessed folder

## b_make_embeddings.py: 
use preprocessed data to create vectorrepresentations of sentences using doc2vec

## c_gicf.py: 
optimize the group-instance cost function, save textlevel results and parameters, uses helperfuctions from 
c1_gradient_helper_functions.py and c2_similarity.py

## d1_bow.py, d2_word_2_vec.py: 
baselinemodels creat model, find parameter for logistic regression on textlevel, save textlevel results and parameter

## e_sentence_level_results.py: 
test gicf and baseline models on sentencelevel test data, save results

## f_make_final_result_tables.py: 
load textlevel and sentence level results, save as cvs tabel

## z_make_sent_pred_in_text.py: 
predict sentiment polarity of each sentence in a text

## z_search_similar.py: 
search for similar sentences according to cosine distance of sentence embeddings


## Thesis:

Abstract

Sentiment analysis is often applied to financial news texts usually
to predict stock market movements. Thereby, researchers focus
their analysis mostly on text level sentiment. Thus, the structure
of sentiment of sentences in financial news texts is unexplored.
Knowing the sentiment polarity of each sentence in a text could
improve the understanding of how financial news affect stock
markets. This work contributes to close this gap by introducing
an approach, which allows to predict the sentiment polarity of
each sentence in financial news texts. The approach is driven
by recent research in the field of natural language processing.
The core is a semi supervised learning algorithm which learns a
sentence level classifier from text level labels. Experiments show,
that the classifier predicts the sentiment of sentences 60% right,
so it outperforms two baseline models.
