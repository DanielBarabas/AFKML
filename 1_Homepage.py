import streamlit as st
import pandas as pd
import data_wrangling as dw

st.set_page_config(page_title="Home page", layout="wide")
st.title("Data Upload page")


user_file = st.file_uploader(
    label="You can upload the data here.", type=["csv", "xlsx"]
)

# TODO add compressed upload option
# TODO frissüljön a dtype a kiírt df-ben - talán st.session_state-tel?
if user_file is not None:
    # TODO downcast numerical values automatically?
    try:
        df = pd.read_csv(user_file)
    except Exception as e:
        print(e)
        df = pd.read_excel(user_file)

    st.write("Here is the head and variable types of your data")
    st.write(pd.concat([df.dtypes.rename("dtype").to_frame().T, df.head(5)]))

    ### Recast datatypes
    vars_recast = st.multiselect("Choose variables to set data types", df.dtypes.index)
    df = dw.cast_dtypes(df, vars_recast)

    # st.write(df.dtypes)
