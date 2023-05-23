import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler


# droping id column
def remove_id(df):
    """
        Drops id column from dataframe.

    :param df: dataframe
    :return: dataframe
    """
    try:
        df = df.drop('id', axis=1)
    except KeyError:
        pass
    return df


# replacing na values
def replace_na(df):
    """
    Replaces Na values with median for numeric columns or mode for object ones.

    :param df: dataframe
    :return: dataframe
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            # fill missing values with mode for categorical variables
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            # fill missing values with median for numerical variables
            df[col].fillna(df[col].median(), inplace=True)
    return df


# outliers removal - replace lower_bound upper_bound
def replace_outliers(data, k=1.5):
    """
    Removes outliers from a pandas DataFrame using the interquartile range (IQR) method.

    :param data: dataframe
    :param k: A multiplier to adjust the outlier threshold. By default, k is set to 1.5.
    :return: dataframe
    """
    data = data.copy()
    for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
        q1, q3 = np.percentile(data[col], [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        mask = (data[col] < lower_bound)
        data.loc[mask, col] = lower_bound
        mask = (data[col] > upper_bound)
        data.loc[mask, col] = upper_bound

    value = data['restecg'].value_counts().idxmax()
    data['restecg'] = data['restecg'].replace(1, value)
    data['restecg'] = data['restecg'].replace(2, 1)

    return data


def remove_overshadowing(df):
    """
    Performs Principal Component Analysis (PCA) on a given dataset.

    :param df: dataframe
    :return:
            A reduced version of the input DataFrame, containing the first n principal components
            where n is the smallest number of components that explain at least 95% of the variance in the data.
            The columns of the new DataFrame are named "PC1", "PC2", etc.

    """

    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(df)

    # Determine the number of principal components to retain
    explained_variances = pca.explained_variance_ratio_
    cumulative_variances = np.cumsum(explained_variances)
    n_components = np.argmax(cumulative_variances > 0.95) + 1

    # Retain the first n principal components
    X_reduced = X_pca[:, :n_components]

    df_reduced = pd.DataFrame(X_reduced, columns=["PC" + str(i) for i in range(1, n_components + 1)])

    return df_reduced


def standarize(df):
    """
        Standarizes dataframe
    :param df: dataframe
    :return: dataframe
    """

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    df = pd.DataFrame(df_scaled, columns=df.columns)
    return df


def normalize(df):
    normalizer = Normalizer()
    df_normalized = normalizer.fit_transform(df)
    df = pd.DataFrame(df_normalized, columns=df.columns)
    return df


def minmax(df):
    minmaxer = MinMaxScaler()
    df_scaled = minmaxer.fit_transform(df)
    df = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)
    return df


def encode_columns(df):
    """
        Encodes categorical column cp (chest pain type)

    :param df: Dataframe
    :return: Encoded dataframe
    """
    df = pd.get_dummies(df, columns=['cp'], prefix='cp', dtype=int)
    df = df.rename(columns={"cp_1": "cp_typ_ang", "cp_2": "cp_atyp_ang", "cp_3": "cp_non_ang", "cp_4": "cp_asympt"})
    return df


def preprocessing(df: pd.DataFrame,
                  rep_out=True, rem_over=True, encode=True, process='stand', handle_na="replace") -> pd.DataFrame:
    """
        Performs preprocessing.

    :param df: Dataframe that is to be transformed.
    :param rep_out: Should outliers be replaced?
    :param rem_over: Should overshadowing variables be replaced?
    :param encode: Should categorical columns be encoded?
    :param process:
        "stand": standarization
        "minmax": minmax scaler
        "norm": normalization
    :param handle_na:
        "drop": removing na values
        "replace": replacing na values with median (numerical) or mode (categorical) depending on the type
    :return: preprocessed dataframe
    """
    # removing id because it holds no information
    df = remove_id(df)

    # handling na
    if handle_na == "drop":
        df = df.dropna(axis=0)
    elif handle_na == "replace":
        df = replace_na(df)

    # replacing outliers
    if rep_out:
        df = replace_outliers(df)

    if encode:
        df = encode_columns(df)

    if process == 'stand':
        df = standarize(df)
    elif process == 'minmax':
        df = minmax(df)
    elif process == 'norm':
        df = normalize(df)

    # removing overshadowing variables
    if rem_over:
        df = remove_overshadowing(df)

    return df
