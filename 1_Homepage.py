import streamlit as st
import pandas as pd
import pages.modules.data_wrangling as dw
from st_aggrid import GridOptionsBuilder, AgGrid

st.set_page_config(page_title="Home page", layout="wide")
st.title("Data Upload page")


user_file = st.file_uploader(
    label="You can upload the data here.", type=["csv", "xlsx"]
)
typelist = ("Numeric", "Categorical")

# TODO add compressed upload option
# TODO what if user wants to upload another data? - session state if statement doesn't allow that (create refresh page button)
if user_file is not None:
    # TODO dtypes makes for bloated df -> downcast numerical values automatically?
    try:
        if "df" not in st.session_state:
            st.session_state["df"] = pd.read_csv(user_file)
    except Exception as e:
        print(e)
        if "df" not in st.session_state:
            st.session_state["df"] = pd.read_excel(user_file)


if "df" in st.session_state:
    st.write("Here are the data types of columns and the first observations")
    # TODO sometimes dtypes in printed df doesn't refresh
    st.write(
        pd.concat(
            [
                st.session_state["df"].dtypes.rename("dtype").to_frame().T,
                st.session_state["df"].head(5),
            ]
        )
    )

    original_types = dw.create_type_df(df=st.session_state["df"])
    orig_dict = dw.create_type_dict(original_types)

    gb = GridOptionsBuilder.from_dataframe(original_types)
    gb.configure_column(
        "Type",
        editable=True,
        cellEditor="agSelectCellEditor",
        cellEditorParams={"values": typelist},
    )

    vgo = gb.build()
    response = AgGrid(
        original_types,
        gridOptions=vgo,
    )

    res_dict = dw.create_type_dict(response.data)

    df, orig_dict = dw.cast_dtype(st.session_state["df"], orig_dict, res_dict)

    st.write(st.session_state["df"].dtypes)
