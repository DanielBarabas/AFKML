import streamlit as st
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
)
import numpy as np


def create_type_df(df: pd.DataFrame, dtype_map: dict):
    original_types = pd.DataFrame(
        {
            "Variable": df.columns.to_list(),
            "Type": [dtype_map[a] for a in [str(typ) for typ in df.dtypes]],
        }
    )
    return original_types


def create_type_dict(type_df: pd.DataFrame):
    type_dict = {}
    for i in range(len(type_df)):
        type_dict[type_df.iloc[i]["Variable"]] = type_df.iloc[i]["Type"]
    return type_dict


def cast_dtype(df: pd.DataFrame, orig_dict: dict, res_dict: dict, dtype_map: dict):
    for key in res_dict.keys():
        if res_dict[key] != orig_dict[key]:
            # Cannot handle exceptions yet
            df[key] = df[key].astype(dtype_map[res_dict[key]])
    orig_dict = res_dict
    return df, orig_dict


######################## Encoding page ##################################


def find_valid_cols(df: pd.DataFrame, target_var: str, dtype_map: dict):
    valid_cols = [
        col
        for col in df.columns.to_list()
        if (
            (dtype_map[str(df[col].dtype)] == "Categorical") and (col not in target_var)
        )
    ]
    return valid_cols


# TODO KILL?
def find_label_type(df, target_var: str, dtype_map: dict):
    y_type = dtype_map[str(df[target_var].dtype)]
    return y_type


def one_hot_encoding(df, x_column):
    encoder = OneHotEncoder(drop="first", sparse_output=False, dtype=np.int8)
    encoded_column = encoder.fit_transform(df[x_column].values.reshape(-1, 1))
    encoded_column_name = [
        x_column + "_one_hot_" + str(cat) for cat in encoder.categories_[0][1:]
    ]
    return encoded_column, encoded_column_name


# TODO Check if multicollinearity can be a problem here
def tartet_encoding(df, x_column, target_column, coltype):
    x = df[x_column].to_numpy().reshape(-1, 1)
    if coltype == "Numeric":
        encoder = TargetEncoder(smooth="auto", target_type="continuous")
        encoded_column = encoder.fit_transform(X=x, y=df[target_column])
        encoded_column_name = x_column + "_target"
    else:
        encoder = TargetEncoder(smooth="auto", target_type="auto")
        encoded_column = encoder.fit_transform(X=x, y=df[target_column])
        if encoder.target_type_ == "binary":
            encoded_column_name = x_column + "_target"
        else:
            encoded_column_name = [
                x_column + "_target_" + str(cat) for cat in encoder.classes_
            ]

    return encoded_column, encoded_column_name


def ordinal_encoding(df, x_column):
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
    encoded_column = oe.fit_transform(df[x_column].values.reshape(-1, 1))
    encoded_column_name = x_column + "_ordinal"
    return encoded_column, encoded_column_name


def create_encoded_column(
    encoding, x_column, df, target_column="", y_type="Categorical"
):
    """
    Use encoder to fit_transform the `original_colname` column and return just the encoded column.
    """
    if encoding == "One-Hot":
        encoded_column, encoded_column_name = one_hot_encoding(df=df, x_column=x_column)
    elif encoding == "Target":
        encoded_column, encoded_column_name = tartet_encoding(
            df, x_column, target_column, coltype=y_type
        )
    elif encoding == "Ordinal":
        encoded_column, encoded_column_name = ordinal_encoding(df, x_column)

    return pd.DataFrame(encoded_column), encoded_column_name


# TODO ez kicsit bonyolultnak tűnik, miért dropolja az y-t, nem lenne egyszerűbb nem hozzáadni
def create_model_df(
    res_df: pd.DataFrame,
    df: pd.DataFrame,
    target_var: str,
    y_type: str,
    dtype_map: dict,
):
    model_df = pd.concat(
        [
            df[col]
            for col in [
                col1
                for col1 in df.columns
                if dtype_map[str(df[col1].dtype)] == "Numeric"
            ]
        ],
        axis=1,
    )
    for _, row in res_df.iterrows():
        encoded_col, encoded_colname = create_encoded_column(
            encoding=row["Encoding"],
            x_column=row["Variable"],
            df=df,
            target_column=target_var,
            y_type=y_type,
        )
        # TODO mit csinál ez az if?
        if isinstance(encoded_colname, str):
            encoded_col.columns = [encoded_colname]
        else:
            encoded_col.columns = encoded_colname

        model_df = pd.concat([model_df, encoded_col], axis=1)
    if target_var in model_df.columns:
        model_df = model_df.drop(target_var, axis=1)
    return model_df
