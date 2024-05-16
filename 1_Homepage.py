import streamlit as st
import pandas as pd

st.set_page_config(page_title="Homepage", page_icon="🏠")
st.title("Data Upload page")
st.sidebar.success("Select a page above.")

user_file = st.sidebar.file_uploader(
    label="You can upload the data here.", type=["csv", "xlsx"]
)

if user_file is not None:
    try:
        df = pd.read_csv(user_file)
    except Exception as e:
        print(e)
        df = pd.read_excel(user_file)

# st.write(df.head(10))
