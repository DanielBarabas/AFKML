import streamlit as st
import pandas as pd


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
