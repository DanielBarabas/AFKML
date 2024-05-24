import streamlit as st
import pandas as pd
from st_aggrid import GridOptionsBuilder, AgGrid
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
)
import numpy as np


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


df = load_data("C:\Projects\Rajk\prog_2\project\prog_machine_project\data\drinking.csv")
# Ezt azért csináltam, hogy meg tudjuk nézni multiclass esetben működik-e
df["weight"] = df["weight"].astype("category")


target_var = st.multiselect(
    "Choose the target variable:", df.columns.to_list(), max_selections=1
)
if len(target_var) > 0:
    target_var = target_var[0]
else:
    target_var = df.columns.to_list()[0]


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

y_type = dtype_map_inverse[str(df[target_var].dtype)]

st.write(target_var)
st.write(y_type)


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


original_encodings = pd.DataFrame({"Variable": valid_cols, "Encoding": "One-Hot"})

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
encoding_dict = {}
for _, row in response.data.iterrows():
    encoding_dict[row[0]] = row[1]

st.write(encoding_dict)

data_list = []
col_list = []


for _, row in response.data.iterrows():
    encoded_col, encoded_colname = create_encoded_column(
        encoding=row[1], x_column=row[0], df=df, target_column=target_var, y_type=y_type
    )
    encoded_col.columns = encoded_colname
    data_list.append(encoded_col)
    col_list.append(encoded_colname)

st.write(col_list)
for data in data_list:
    st.write(data)
