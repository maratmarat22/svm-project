import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


# noinspection PyPep8Naming
def preprocess(dataset_file, sep, target_column):
    df = pd.read_csv(dataset_file, sep=sep)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    cat_cols += [col for col in X.columns if X[col].nunique() < 10 and X[col].dtype in ["int64", "float64"]]

    if cat_cols:
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        X_cat = pd.DataFrame(encoder.fit_transform(X[cat_cols]), columns=encoder.get_feature_names_out())
        X = X.drop(columns=cat_cols)
        X = pd.concat([X, X_cat], axis=1)

    if y.dtype == "object" or y.nunique() < 10:  # Если метки классов нечисловые
        le = LabelEncoder()
        y = le.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y
