import numpy as np
from sklearn.metrics import confusion_matrix as cm

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    return true_positives / (true_positives + false_positives + 1e-10)

def recall(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    return true_positives / (true_positives + false_negatives + 1e-10)

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec + 1e-10)

def confusion_matrix(y_true, y_pred):
    tn, fp, fn, tp = cm(y_true, y_pred).ravel()
    return {
        'true negatives': tn,
        'false positives': fp,
        'false negatives': fn,
        'true positives': tp,
    }