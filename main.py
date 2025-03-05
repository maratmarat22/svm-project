import numpy as np
from sklearn.model_selection import train_test_split
from my_svm.preprocessing import preprocess
from my_svm.support_vector_machine import LinearSVM, KernelSVM
from my_svm import metrics
from my_svm import balancing

# noinspection PyPep8Naming
def main(dataset_file, sep, target_column, balance_pref='none', SVM_type='linear', random_state=0):
    X, y = preprocess(dataset_file, sep, target_column)
    #X, y = load_iris(return_X_y=True, as_frame=False)
    print_unique(y, 'classes counts:')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)
    print_unique(y_train, 'train classes counts:')

    match balance_pref:
        case 'undersampling':
            X_train, y_train = balancing.undersample(X_train, y_train, random_state=random_state)
        case 'oversampling':
            X_train, y_train = balancing.oversample(X_train, y_train, random_state=random_state)
        case 'smote':
            X_train, y_train = balancing.smote(X_train, y_train, random_state=random_state)
        case 'none':
            pass
        case _:
            raise ValueError(f'no such balance pref: {balance_pref}')

    print_unique(y_train, f'train classes counts after balancing ({balance_pref}):')

    if SVM_type == 'linear':
        svm = LinearSVM(epochs=500, C=0.1, learning_rate=0.001, random_state=random_state)
    elif SVM_type == 'kernel':
        svm = KernelSVM(epochs=10, C=0.1, learning_rate=0.01, gamma=0.1, random_state=random_state)
    else:
        raise ValueError(f'no such SVM type: {SVM_type}')

    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    # Вывод метрик
    print('accuracy:', metrics.accuracy(y_test, y_pred))
    print('precision:', metrics.precision(y_test, y_pred))
    print('recall:', metrics.recall(y_test, y_pred))
    print('f1 score:', metrics.f1_score(y_test, y_pred))
    print('confusion matrix:')
    cm = metrics.confusion_matrix(y_test, y_pred)
    for key, val in cm.items():
        print(f'\t{key}: {val}')

def print_unique(y, msg):
    print(msg)
    classes, counts = np.unique(y, return_counts=True)
    for cls, count in zip(classes, counts):
        print(f'{cls}: {count}')

if __name__ == '__main__':
    g_dataset_file = 'data/bank.csv'
    g_sep = ';'
    g_target_column = 'y'
    g_balance_pref = 'smote' # undersampling / oversampling / smote / none (default)
    g_SVM_type = 'linear'  # linear (default) / kernel
    g_random_state = 42
    main(g_dataset_file, g_sep, g_target_column, g_balance_pref, g_SVM_type, g_random_state)
