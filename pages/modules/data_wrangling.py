import streamlit as st
import pandas as pd

# Not used anymore
def cast_dtypes(df: pd.DataFrame, vars_recast: list) -> pd.DataFrame:
    """Recasts dtypes of df with using streamlit dropdowns menus

    Args:
        df (pd.DataFrame): df uploaded by user
        vars_recast (list): selected variables to recast

    Returns:
        pd.DataFrame: df with recasted columns
    """
    dtype_map = {"Numeric": float, "Categorical": "category"}
    vars_recast = {var: None for var in vars_recast}

    # TODO add more var types: ordinal, date-time, string -> do proper encoding
    for var in vars_recast.keys():
        vars_recast[var] = st.selectbox(
            f"Select dtype for {var}", ["Numeric", "Categorical"]
        )

    for column, dtype in vars_recast.items():
        df[column] = df[column].astype(dtype_map[dtype])

    return df

# Are these dictionaries OK here?
dtype_map = {"Numeric": float, "Categorical": "category"}
dtype_map_inverse = {
    "float64": "Numeric",
    "category": "Categorical",
    "object": "Categorical",
    "int64": "Numeric",
}

def create_type_df(df: pd.DataFrame):
    original_types = pd.DataFrame(
    {
        "Variable": df.columns.to_list(),
        "Type": [dtype_map_inverse[a] for a in [str(typ) for typ in df.dtypes]],
        }
    )
    return original_types

def create_type_dict(type_df: pd.DataFrame):
    type_dict = {}
    for i in range(len(type_df)):
        type_dict[type_df.iloc[i]["Variable"]] = type_df.iloc[i]["Type"]
    return type_dict

def cast_dtype(df: pd.DataFrame, orig_dict: dict, res_dict: dict):
    for key in res_dict.keys():
        if res_dict[key] != orig_dict[key]:
            # Cannot handle exceptions yet
            df[key] = df[key].astype(dtype_map[res_dict[key]])
    orig_dict = res_dict
    return df, orig_dict