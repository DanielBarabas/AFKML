import streamlit as st
import pandas as pd
from st_aggrid import GridOptionsBuilder, AgGrid
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, TargetEncoder
import numpy as np


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


df = load_data("C:\Projects\Rajk\prog_2\project\prog_machine_project\data\drinking.csv")

target_var = st.multiselect(
    "Choose the target variable:", df.columns.to_list(), max_selections=1
)
if len(target_var) > 0:
    target_var = target_var[0]
else:
    target_var = df.columns.to_list()[0]
st.write(target_var)


def create_encoded_column(encoder, original_colname, df):
    """
    Use encoder to fit_transform the `original_colname` column and return just the encoded column.
    """
    # you fit on training data
    encoded_column = encoder.fit_transform(df[original_colname].values.reshape(-1, 1))
    return encoded_column


def encode_target(target_var: str, df: pd.DataFrame, encoding: str = "One-hot"):
    if encoding == "One-hot":
        encoder = OneHotEncoder(drop="first", sparse_output=False, dtype=np.int64)
    elif encoding == "Target":
        pass
    else:
        pass
    return create_encoded_column(encoder, target_var, df)


target_encoded = encode_target(target_var, df)

encodings = ("One-Hot", "Target", "Ordinal")
dtype_map = {"Numeric": float, "Categorical": "category"}
dtype_map_inverse = {
    "float64": "Numeric",
    "category": "Categorical",
    "object": "Categorical",
    "int64": "Numeric",
}
valid_cols = [
    col
    for col in df.columns.to_list()
    if (
        (dtype_map_inverse[str(df[col].dtype)] == "Categorical")
        and (col not in target_var)
    )
]

original_encodings = pd.DataFrame({"Variable": valid_cols, "Encoding": "Ordinal"})

gb = GridOptionsBuilder.from_dataframe(original_encodings)
gb.configure_column(
    "Encoding",
    editable=True,
    cellEditor="agSelectCellEditor",
    cellEditorParams={"values": encodings},
)

vgo = gb.build()
response = AgGrid(
    original_encodings,
    gridOptions=vgo,
)
