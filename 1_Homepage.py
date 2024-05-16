import streamlit as st
import pandas as pd
import data_wrangling as dw

st.set_page_config(page_title="Home page", layout="wide")
st.title("Data Upload page")


user_file = st.file_uploader(
    label="You can upload the data here.", type=["csv", "xlsx"]
)

# TODO add compressed upload option
if user_file is not None:
    # TODO downcast numerical values automatically?
    try:
        if "df" not in st.session_state:
            st.session_state["df"] = pd.read_csv(user_file)
    except Exception as e:
        print(e)
        if "df" not in st.session_state:
            st.session_state["df"] = pd.read_excel(user_file)

    st.write("Here are the data types of columns and the first observations")
    # TODO printed dtypes should refresh (using session_state is not enough)
    st.write(
        pd.concat(
            [
                st.session_state["df"].dtypes.rename("dtype").to_frame().T,
                st.session_state["df"].head(5),
            ]
        )
    )

    ### Recast datatypes
    vars_recast = st.multiselect(
        "Choose variables to set data types", st.session_state["df"].dtypes.index
    )
    df = dw.cast_dtypes(st.session_state["df"], vars_recast)
