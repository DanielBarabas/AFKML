import streamlit as st
import pandas as pd

st.set_page_config(page_title="Homepage", layout="wide", page_icon="🏠")
st.title("Data Upload page")


user_file = st.file_uploader(
    label="You can upload the data here.", type=["csv", "xlsx"]
)

# TODO add compressed upload option
if user_file is not None:
    try:
        df = pd.read_csv(user_file)
    except Exception as e:
        print(e)
        df = pd.read_excel(user_file)

    dtypes = df.dtypes

    head_dtypes = pd.concat([dtypes.rename("dtype").to_frame().T, df.head(5)])

    st.write("Here is the head and variable types of your data")
    st.write(head_dtypes)

    vars_recast = st.multiselect("Choose variables to set data types", dtypes.index)
    vars_recast = {var: None for var in vars_recast}
    # st.write(vars_recast)

    # TODO add more var types: ordinal, date-time, string -> do proper encoding
    for var in vars_recast.keys():
        # Display dropdown for each selected variable
        vars_recast[var] = st.selectbox(
            f"Select dtype for {var}", ["Numeric", "Categorical"]
        )

    st.write(df.dtypes)
