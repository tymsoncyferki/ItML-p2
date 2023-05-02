import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('heart_disease_patients.csv')

# droping id column
def remove_id(df):
    df.drop('id', axis=1, inplace=True)

# replacing na values
def replace_na(df):
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
    # TODO: Which columns to replace outliers from
    for col in data.columns:
        q1, q3 = np.percentile(data[col], [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        mask = (data[col] < lower_bound) 
        data.loc[mask, col] = lower_bound
        mask = (data[col] > upper_bound)
        data.loc[mask, col] = upper_bound
    return data

def remove_overshadowing(df):    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Determine the number of principal components to retain
    explained_variances = pca.explained_variance_ratio_
    cumulative_variances = np.cumsum(explained_variances)
    n_components = np.argmax(cumulative_variances > 0.95) + 1

    # Retain the first n principal components
    X_reduced = X_pca[:, :n_components]

    df_reduced = pd.DataFrame(X_reduced, columns=["PC"+str(i) for i in range(1, n_components+1)])
    
    return df_reduced


def preprocessing(df, rep_out=True, rem_over = True, stand = True, handle_na="replace"):
    """
    Performs preprocessing
    Parameters:
    df (dataframe): Dataframe that is to be transformed.
    rep_out (boolean): Should outliers be replaced?
    rem_over (boolean): Should overshadowing variables be replaced?
    stand (boolean): Should dataframe be standarized?
    handle_na (string): 
        "drop": removing na values
        "replace": replacing na values with median (numerical) or mode (categorical) depending on the type

    Returns:
    dataframe: preprocessed dataframe
    """
    # removing id because it holds no information
    remove_id(df)

    # handling na
    if handle_na == "drop":
        df.dropna(axis=0, inplace = True)
    elif handle_na == "replace":
        df = replace_na(df)

    # replacing outliers
    if rep_out:
        df = replace_outliers(df)

    # removing overshadowing variables 
    if rem_over:
        df = remove_overshadowing(df)

    return df
    