import streamlit as st
import pandas as pd
import modules.data_wrangling as dw
from st_aggrid import GridOptionsBuilder, AgGrid


dtype_map = {"Numeric": float, "Categorical": "category"}
dtype_map_inverse = {
    "float64": "Numeric",
    "category": "Categorical",
    "object": "Categorical",
    "int64": "Numeric",
    "datetime64[ns]": "Date",
}


st.set_page_config(page_title="Home page", layout="wide")
st.title("Data Upload page")


user_file = st.file_uploader(
    label="You can upload the data here.", type=["csv", "xlsx"]
)
typelist = ("Numeric", "Categorical", "Date")

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
    st.write("Here are the first rows of your data frame")
    st.write(st.session_state["df"].head(5))

    original_types = dw.create_type_df(st.session_state["df"], dtype_map_inverse)
    orig_dict = dw.create_type_dict(original_types)

    gb = GridOptionsBuilder.from_dataframe(original_types)
    gb.configure_column(
        "Type",
        editable=True,
        cellEditor="agSelectCellEditor",
        cellEditorParams={"values": typelist},
    )

    st.header("Change data types")
    st.write(
        "Double click on the cells under Type column to change the data type of the variable"
    )
    vgo = gb.build()
    response = AgGrid(
        original_types,
        gridOptions=vgo,
    )

    res_dict = dw.create_type_dict(response.data)
    try:
        df, orig_dict = dw.cast_dtype(
            st.session_state["df"], orig_dict, res_dict, dtype_map
        )
    except:
        st.write("Not a valid type!")
