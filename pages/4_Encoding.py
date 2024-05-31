import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from st_aggrid import GridOptionsBuilder, AgGrid
from modules.data_wrangling import find_valid_cols, find_label_type, create_model_df


dtype_map_inverse = {
    "float64": "Numeric",
    "category": "Categorical",
    "object": "Categorical",
    "int64": "Numeric",
}

with st.expander(label="Target variable selection", expanded=True):
    target_var = st.selectbox(
        "Choose the target variable for your analysis:",
        st.session_state["df"].columns.to_list(),
        help="The target variable has to be encoded in a different way than features",
    )
    st.session_state["y_colname"] = target_var

    # Encode target variable and specify problem type
    object_or_cat = pd.api.types.is_object_dtype(
        st.session_state["df"][target_var]
    ) or pd.api.types.is_categorical_dtype(st.session_state["df"][target_var])

    if object_or_cat:
        label_encoder = LabelEncoder()
        st.session_state["y"] = label_encoder.fit_transform(
            st.session_state["df"][target_var]
        )
        st.session_state["le"] = label_encoder

        n_unique_cat = st.session_state["df"][target_var].nunique()
        if n_unique_cat == 2:
            st.session_state["problem_type"] = "Binary classification"
            st.session_state["y_type"] = "categorical with two categories"
        else:
            st.session_state["problem_type"] = "Multiclass classification"
            st.session_state["y_type"] = "categorical with multiple categories"
    else:
        # cast to pd.DataFrame otherwise sklearn models doesn't run
        st.session_state["y"] = pd.DataFrame(st.session_state["df"][target_var])
        st.session_state["problem_type"] = "Regression"
        st.session_state["y_type"] = "numeric"

    st.write(
        f'You are going to do {st.session_state["problem_type"]} since the target variable, {st.session_state["y_colname"]} is {st.session_state["y_type"]}'
    )


encodings = ("One-Hot", "Target", "Ordinal")
valid_cols = find_valid_cols(st.session_state["df"], target_var, dtype_map_inverse)
y_type = find_label_type(st.session_state["df"], target_var, dtype_map_inverse)

st.write(target_var, y_type)

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

st.write(response.data)

st.session_state["X"] = create_model_df(
    response.data, st.session_state["df"], target_var, y_type, dtype_map_inverse
)
st.write(st.session_state["X"])
# This has the encoded features in it
st.write(st.session_state["df"].shape)
st.write(st.session_state["X"].shape)
