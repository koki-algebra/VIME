import pandas as pd
from sklearn.preprocessing import LabelEncoder


def one_hot_encoding(df: pd.DataFrame, target_column_name: str) -> pd.DataFrame:
    # features dataframe
    X_df = df.drop(labels=target_column_name, axis=1)
    # label series
    y_sr = df[target_column_name]

    # feature one-hot encoding
    X_df = pd.get_dummies(data=X_df)

    # label encoding
    encoder = LabelEncoder()
    encoder.fit(y_sr)
    encoded = encoder.transform(y_sr)
    y_sr = pd.Series(encoded, name=target_column_name).astype("int64")

    return pd.concat([X_df, y_sr], axis=1)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    categorical = df.select_dtypes(include="object")
    min_df = df.min(numeric_only=True)
    max_df = df.max(numeric_only=True)
    normalized = (df.select_dtypes(exclude="object") - min_df) / (max_df - min_df)

    return pd.concat([normalized, categorical], axis=1)
