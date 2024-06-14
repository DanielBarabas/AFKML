import pandas as pd
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
)
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import altair as alt


def create_type_df(df: pd.DataFrame, dtype_map: dict) -> pd.DataFrame:
    """Creates a dataframe that assigns our variable categories (Numeric, Categorical) to each variable in the dataframe

    Args:
        pd.DataFrame: dataframe of the data
        dict: dictionary that assigns our categories to the possible dtypes


    Returns:
        pd.DataFrame: Each row is a variable in the data and our categories are assigned to each variable
    """
    original_types = pd.DataFrame(
        {
            "Variable": df.columns.to_list(),
            "Type": [dtype_map[a] for a in [str(typ) for typ in df.dtypes]],
        }
    )
    return original_types


def create_type_dict(type_df: pd.DataFrame) -> dict:
    """Creates a dictionary

    Args:
        type_df (pd.DataFrame):

    Returns:
        dict: Keys are the variables in the data, values are their categories (Numeric, Categorical)
    """
    type_dict = {}
    for i in range(len(type_df)):
        type_dict[type_df.iloc[i]["Variable"]] = type_df.iloc[i]["Type"]
    return type_dict


def cast_dtype(
    df: pd.DataFrame, orig_dict: dict, res_dict: dict, dtype_map: dict
) -> tuple:
    """_summary_

    Args:
        df (pd.DataFrame): data
        orig_dict (dict): previous type dictionary
        res_dict (dict): current type dictionary
        dtype_map (dict): maps dtypes to our types (Numeric, Categorical)

    Returns:
        tuple: first element is the updated dataframe, second element is the new previous dictionary
    """
    for key in res_dict.keys():
        if res_dict[key] != orig_dict[key]:
            df[key] = df[key].astype(dtype_map[res_dict[key]])
    orig_dict = res_dict
    return df, orig_dict


######################## Encoding page ##################################


@st.cache_data
def find_cat_cols(df: pd.DataFrame, target_var: str, dtype_map: dict) -> tuple:
    """_summary_

    Args:
        df (pd.DataFrame): data
        target_var (str): y
        dtype_map (dict): maps dtypes to our types (Numeric, Categorical)

    Returns:
        list: list of columns to encode (categorical and not target variables)
    """
    valid_cols = [
        col
        for col in df.columns.to_list()
        if (
            (dtype_map[str(df[col].dtype)] == "Categorical") and (col not in target_var)
        )
    ]
    return valid_cols


def find_cont_cols(df,target_var):
    numeric_cols = df.select_dtypes(include=[np.number])
    # Filter out columns with only 0s and 1s
    non_binary_numeric_cols = numeric_cols.loc[
        :,
        ~(numeric_cols.nunique() == 2)
        & ~((numeric_cols == 0) | (numeric_cols == 1)).all(),
    ]
    cont_cols = [col for col in non_binary_numeric_cols.columns.tolist() if col != target_var]
    
    return cont_cols


# TODO KILL?
def find_label_type(df: pd.DataFrame, target_var: str, dtype_map: dict) -> str:
    """Returns "Numeric" or "Categorical" for the target variable. This is needed for the target encoding.

    Args:
        df (pd.DataFrame):
        target_var (str): y
        dtype_map (dict): maps dtypes to our types (Numeric, Categorical)

    Returns:
        str: "Numeric" or "Categorical"
    """
    y_type = dtype_map[str(df[target_var].dtype)]
    return y_type


############################### Encoding ##################################
@st.cache_data
def one_hot_encoding(df: pd.DataFrame, x_column: str) -> tuple:
    """Encodes one column using onehot encoding

    Args:
        df (pd.DataFrame): data
        x_column (str): column to encode

    Returns:
        tuple: first element is the encoded column (numpy array), second element is the encoded name
    """
    encoder = OneHotEncoder(drop="first", sparse_output=False, dtype=np.int8)
    encoded_column = encoder.fit_transform(df[x_column].values.reshape(-1, 1))
    encoded_column_name = [
        x_column + "_one_hot_" + str(cat) for cat in encoder.categories_[0][1:]
    ]
    return encoded_column, encoded_column_name


@st.cache_data
def tartet_encoding(
    df: pd.DataFrame, x_column: str, target_column: str, coltype: str
) -> tuple:
    """Encodes one column using target encoding

    Args:
        df (pd.DataFrame): data
        x_column (str): variable to encode
        target_column (str): y variable
        coltype (str): y type (Numeric or Categorical)

    Returns:
        tuple: first element is the encoded column (numpy array or multidimensional array in case of multiclass prediction), second element is the encoded name (or a list of them in case of multiclass prediction)
    """
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


@st.cache_data
def ordinal_encoding(df: pd.DataFrame, x_column: str) -> tuple:
    """Encodes one column using ordinal encoding

    Args:
        df (pd.DataFrame): data
        x_column (str): column to encode

    Returns:
        tuple: first element is the encoded column (numpy array), second element is the encoded name
    """
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
    encoded_column = oe.fit_transform(df[x_column].values.reshape(-1, 1))
    encoded_column_name = x_column + "_ordinal"
    return encoded_column, encoded_column_name


@st.cache_data
def create_encoded_column(
    encoding: str, x_column: str, df: pd.DataFrame, target_column: str, y_type: str
) -> tuple:
    """Encodes the given column using according to the users chosen encoding

    Args:
        encoding (str): Encoding type (One-hot, Ordinal or Target)
        x_column (str): Variable to encode
        df (pd.DataFrame): data
        target_column (str): y
        y_type (str): y type (Numeric or Categorical)

    Returns:
        tuple: first element is the encoded column (pd.DataFrame), second element is the encoded variable name
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


@st.cache_data
def create_cat_df(
    res_df: pd.DataFrame,
    df: pd.DataFrame,
    target_var: str,
    y_type: str,
    dtype_map: dict,
) -> pd.DataFrame:
    """Creates the encoded X (feature) DataFrame

    Args:
        res_df (pd.DataFrame): user input (variable name, encoding type)
        df (pd.DataFrame): data
        target_var (str): y
        y_type (str): y type (Numeric, Categorical)
        dtype_map (dict): maps dtypes to our types (Numeric, Categorical)

    Returns:
        pd.DataFrame: Encoded feature (X) DataFrame
    """
    model_df = pd.DataFrame()
    """model_df = pd.concat(
        [
            df[col]
            for col in [
                col1
                for col1 in df.columns
                if dtype_map[str(df[col1].dtype)] == "Numeric"
            ]
        ],
        axis=1,
    )"""
    for _, row in res_df.iterrows():
        encoded_col, encoded_colname = create_encoded_column(
            encoding=row["Encoding"],
            x_column=row["Variable"],
            df=df,
            target_column=target_var,
            y_type=y_type,
        )
        if isinstance(encoded_colname, str):
            encoded_col.columns = [encoded_colname]
        else:
            encoded_col.columns = encoded_colname

        model_df = pd.concat([model_df, encoded_col], axis=1)
    if target_var in model_df.columns:
        model_df = model_df.drop(target_var, axis=1)
    return model_df


###################### Continous feature engineering ###############
def log_column(df, column):
    """
    Transform the given column to its logarithm.
    """
    col = np.log(df[column])
    return col


def standardize_column(df, column):
    """
    Standardize the given column.
    """
    mean = df[column].mean()
    std = df[column].std()
    col = (df[column] - mean) / std
    return col


def cut_column(df, column, lower_percent=None, upper_percent=None):
    """
    Cut the values in the given column based on the lower and upper bounds.
    If a number is below the lower bound, set it to the lower bound.
    If a number is above the upper bound, set it to the upper bound.
    """

    if lower_percent is not None:
        lower_bound = df[column].quantile(lower_percent / 100)
        col = np.where(df[column] < lower_bound, lower_bound, df[column])

    # Calculate the upper bound as a percentage of the column values
    if upper_percent is not None:
        upper_bound = df[column].quantile(upper_percent / 100)
        col = np.where(df[column] > upper_bound, upper_bound, df[column])
    return col


@st.cache_data
def create_cont_df(original_df, cont_cols, cont_res_df, cut_size):
    cont_df = original_df[cont_cols]
    for _, row in cont_res_df.iterrows():
        if row["Transformation"] == "None":
            cont_df[row["Variable"]] = original_df[row["Variable"]]
        elif row["Transformation"] == "Log":
            if cont_df[row["Variable"]].min() <= 0:
                pass
            else:
                cont_df[row["Variable"]] = log_column(cont_df, row["Variable"])

        elif row["Transformation"] == "Standarize":
            cont_df[row["Variable"]] = standardize_column(cont_df, row["Variable"])

        elif row["Transformation"] == "Cut":
            cont_df[row["Variable"]] = cut_column(
                cont_df, row["Variable"], cut_size[0], cut_size[1]
            )

    return cont_df


@st.cache_data
def create_x_df(cont_df, cat_df):
    return pd.concat([cont_df, cat_df], axis=1)


################### PCA ########################


@st.cache_data
def filter_data(df, cols):
    return df[cols]


@st.cache_data
def create_pca_before(df):
    pipeline = Pipeline([("scaler", StandardScaler()), ("pca", PCA())])
    pipeline.fit(df)

    pca = pipeline.named_steps["pca"]
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = explained_variance_ratio.cumsum()
    variance_df = pd.DataFrame(
        {
            "Number of Principal Components": range(
                1, len(explained_variance_ratio) + 1
            ),
            "Explained Variance Ratio": explained_variance_ratio,
            "Cumulative Explained Variance": cumulative_explained_variance,
        }
    )
    return variance_df


@st.cache_data
def create_pca_after(df, n_comp):
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("pca", PCA(n_components=n_comp))]
    )
    principal_components = pipeline.fit_transform(df)
    principal_df = pd.DataFrame(
        data=principal_components, columns=[f"PC{i+1}" for i in range(n_comp)]
    )
    return principal_df


@st.cache_resource(experimental_allow_widgets=True)
def create_pca_plots(df):
    explained_variance_chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x="Number of Principal Components",
            y="Explained Variance Ratio",
            tooltip=["Number of Principal Components", "Explained Variance Ratio"],
        )
        .properties(title="Explained Variance Ratio by Principal Components")
    )

    # Plotting the cumulative explained variance using Altair
    cumulative_variance_chart = (
        alt.Chart(df)
        .mark_line(point=True, color="orange")
        .encode(
            x="Number of Principal Components",
            y="Cumulative Explained Variance",
            tooltip=["Number of Principal Components", "Cumulative Explained Variance"],
        )
        .properties(title="Cumulative Explained Variance by Principal Components")
    )

    # Combine the two charts
    combined_chart = alt.hconcat(explained_variance_chart, cumulative_variance_chart)
    return combined_chart


############################# EDA ############################################
@st.cache_data
def cast_date_to_timestamp(df: pd.DataFrame):
    for col in df.columns:
        if df[col].dtype == "datetime64[ns]":
            df[col] = df[col].astype("int64")
    return df
