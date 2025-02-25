import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess(data: DataFrame, target_column: str):
    data_encoded = pd.get_dummies(data, drop_first=True)

    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(data_encoded.drop(columns=[target_column]).values)

    labels = np.where(data_encoded[target_column] == np.unique(data_encoded[target_column])[0], -1, 1)

    return feature_matrix, labels
