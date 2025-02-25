import numpy as np

def accuracy(l_true, l_pred):
    return np.mean(l_true == l_pred)

def f1_score(l_true, l_pred):
    true_positives = np.sum((l_true == 1) & (l_pred == 1))
    false_positives = np.sum((l_true == -1) & (l_pred == 1))
    false_negatives = np.sum((l_true == 1) & (l_pred == -1))
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    return 2 * (precision * recall) / (precision + recall)