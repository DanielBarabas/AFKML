import streamlit as st
import pandas as pd
from modelling import train_test_X_y_split

# TODO töröl modelling a pages mappából

st.set_page_config(page_title="Modelling", layout="wide")


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df[["sex", "DRK_YN"]] = df[["sex", "DRK_YN"]].astype("category")
    return df


df = load_data(
    "C:/Users/aronn/OneDrive/Asztali gép/Rajk/Prog 2/prog_machine_project/data/smoking_drinking.csv"
)

st.write(df.head(10))

with st.expander(label="Preparation", expanded=True):
    test_ratio = st.slider(
        "Select what ratio you want the test set to be",
        min_value=0.1,
        max_value=0.9,
        value=0.2,
        help="Normally test set ratio is between 0.1 and 0.3",
    )

    X_train, X_test, y_train, y_test = train_test_X_y_split(
        df=df, y_colname="DRK_YN", test_ratio=test_ratio
    )

    st.write(len(X_train), len(X_test), len(y_train), len(y_test))

    model_type = st.selectbox("Select model", options=["Random forest"])


with st.expander(label="Parameters"):
    st.write("shit")
