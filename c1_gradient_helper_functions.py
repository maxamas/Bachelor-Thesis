import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_over_array(x, beta):
    return sigmoid(np.dot(x, beta))

def similarity_derivat(y, y_der_matrix, W_ij):
    y_diff = get_diff(y)
    a = np.dot(2 * np.multiply(W_ij, y_diff).T, y_der_matrix)
    b = np.dot(2 * np.multiply(W_ij, y_diff), y_der_matrix)
    return np.sum(a, axis=0) - np.sum(b, axis=0)

def get_diff(y):
    y = y.reshape((-1, 1))
    d = np.subtract(y, y.T)
    return 2 * d

def y_derivat(y, x):
    y_der = np.multiply(y, (1 - y))
    a = y_der.reshape((-1, 1))
    return a * x

def group_derivative(y, y_der, group_scores, group_lengths):
    (_, columns) = y_der.shape
    sum_array = np.zeros(columns)

    for i in range(len(group_scores)):
        if group_lengths[i] <= 0:  # if length > 0
            continue
        group_score = group_scores[i]
        frm = np.sum(group_lengths[0:i])
        to = frm + group_lengths[i]
        a = 2 * (np.average(y[frm:to]) - group_score * 2 * 0.5)
        b = (np.sum(y_der[frm:to, :], axis=0)) / group_lengths[i]
        sum_array += np.multiply(a, b)
    return sum_array

def loss(similarity, sigmoid_values, group_scores, group_lengths, lam):
    sum_array = 0
    
    diff_sigmoid_values = get_diff(sigmoid_values)
    sq_diff_sigmoid_values = diff_sigmoid_values ** 2
    term_1 = similarity * sq_diff_sigmoid_values
    term_1 = np.sum(term_1, axis=0)
    term_1 = np.sum(term_1, axis=0) / group_scores.shape[0]
    
    for i in range(len(group_scores)):
        if group_lengths[i] <= 0:  # if length > 0
            continue
        group_score = group_scores[i]
        frm = np.sum(group_lengths[0:i])
        to = frm + group_lengths[i]
        
        a = (np.average(sigmoid_values[frm:to]) - group_score) ** 2
        sum_array += a
        
    term_2 = sum_array / group_scores.shape[0] * lam
    
    return term_1 + term_2
    
    
    
