from collections import Counter
import numpy as np
from imblearn.over_sampling import SMOTE

# noinspection PyPep8Naming
def undersample(X, y, random_state=0):
    class_counts = Counter(y)
    min_count = min(class_counts.values())

    X_resampled, y_resampled = [], []

    np.random.seed(random_state)

    for label in class_counts:
        indices = np.where(y == label)[0]
        selected_indices = np.random.choice(indices, min_count, replace=False)
        X_resampled.extend(X[selected_indices])
        y_resampled.extend(y[selected_indices])

    return np.array(X_resampled), np.array(y_resampled)

# noinspection PyPep8Naming
def oversample(X, y, random_state=0):
    class_counts = Counter(y)
    max_count = max(class_counts.values())

    X_resampled, y_resampled = [], []

    np.random.seed(random_state)

    for label in class_counts:
        indices = np.where(y == label)[0]
        selected_indices = np.random.choice(indices, max_count, replace=True)
        X_resampled.extend(X[selected_indices])
        y_resampled.extend(y[selected_indices])

    return np.array(X_resampled), np.array(y_resampled)

# noinspection PyPep8Naming
def smote(X, y, random_state=0):
    smote_instance = SMOTE(random_state=random_state)
    return smote_instance.fit_resample(X, y)