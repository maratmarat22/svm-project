from pandas import read_csv
from sklearn.model_selection import train_test_split
from my_svm.preprocessing import preprocess
from my_svm.support_vector_machine import SupportVectorMachine
from my_svm import metrics
from my_svm.balancing import undersample


import numpy as np

def main(dataset_file: str, sep: str, target_column: str, balance_pref: str) -> None:
    dataset = read_csv(f'data/{dataset_file}', sep=sep)
    feature_matrix, labels = preprocess(dataset, target_column=target_column)

    _, counts = np.unique(labels, return_counts=True)
    print(f'class counts before balancing: -1 -> {counts[0]}; 1 -> {counts[1]}')

    fm_train, fm_test, l_train, l_test = train_test_split(feature_matrix, labels, test_size=0.25, random_state=27)

    if balance_pref == 'undersampling':
        fm_train, l_train = undersample(fm_train, l_train)

    _, counts = np.unique(l_train, return_counts=True)
    print(f'train class counts after balancing: -1 -> {counts[0]}; 1 -> {counts[1]}')

    svm = SupportVectorMachine(learning_rate=0.0001, lambda_param=0.01, n_iters=1)
    svm.fit(fm_train, l_train)
    l_pred = svm.predict(fm_test)

    print('accuracy:', metrics.accuracy(l_test, l_pred))
    print('F1 score:', metrics.f1_score(l_test, l_pred))

if __name__ == '__main__':
    g_dataset_file = 'adult.data' # 'g' stands for 'global'
    g_sep = ','
    g_target_column = 'label_ >50K'
    g_balance_pref = 'none' # undersampling / oversampling / none (default)
    main(g_dataset_file, g_sep, g_target_column, g_balance_pref)
