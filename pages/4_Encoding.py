import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from st_aggrid import GridOptionsBuilder, AgGrid
from modules.data_wrangling import (
    find_cat_cols,
    find_label_type,
    create_cat_df,
    filter_data,
    create_pca_before,
    create_pca_after,
    create_pca_plots,
    find_cont_cols,
    create_cont_df,
    create_x_df,
)


dtype_map_inverse = {
    "float64": "Numeric",
    "category": "Categorical",
    "object": "Categorical",
    "int64": "Numeric",
    "datetime64[ns]": "Date",
}
encodings = ("One-Hot", "Target", "Ordinal")


st.set_page_config(page_title="Encoding", layout="wide")

# Don't run until no data is uploaded
if "df" not in st.session_state:
    st.write("First upload some data on the homepage")
    st.stop()


st.title("Feature Engineering")


st.header("Choose target variable")
st.write(
    "Keep in mind that any changes here automatically removes all principal components from your data!"
)
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


cat_cols = find_cat_cols(st.session_state["df"], target_var, dtype_map_inverse)
y_type = find_label_type(st.session_state["df"], target_var, dtype_map_inverse)
original_encodings = pd.DataFrame({"Variable": cat_cols, "Encoding": "One-Hot"})


st.header("Encode categorical variables")
st.write(
    "Keep in mind that any changes here automatically removes all principal components from your data!"
)
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
cat_res_df = response.data
# Create X df
cat_df = create_cat_df(
    cat_res_df, st.session_state["df"], target_var, y_type, dtype_map_inverse
)
st.header("Transfrorm continous variables")
st.write(
    "Keep in mind that any changes here automatically removes all principal components from your data!"
)
cont_cols = find_cont_cols(st.session_state["df"])

cut_size = st.slider(
    "Select the upper and lower percentiles that you want to cut for the chosen variables",
    min_value=0,
    max_value=100,
    value=[1, 99],
)

settings = ("None", "Cut", "Log", "Standarize")
original_settings = pd.DataFrame({"Variable": cont_cols, "Transformation": "None"})


gb = GridOptionsBuilder.from_dataframe(original_settings)
gb.configure_column(
    "Transformation",
    editable=True,
    cellEditor="agSelectCellEditor",
    cellEditorParams={"values": settings},
)

vgo = gb.build()
response = AgGrid(
    original_settings,
    gridOptions=vgo,
)

cont_res_df = response.data
cont_df = create_cont_df(st.session_state["df"], cont_cols, cont_res_df, cut_size)
st.session_state["X"] = create_x_df(cont_df, cat_df)
st.write(st.session_state["X"])

if st.toggle(label="PCA"):
    variance_df = None
    feat_used = st.multiselect(
        "Choose the features to be included in PCA",
        options=st.session_state["X"].columns,
        default=st.session_state["X"].columns.to_list(),
    )
    if st.toggle(
        label="Show PCA plots (Turn off while choosing the variables if computation is too slow)",
        value=False,
    ):
        # df_for_dimred = filter_data(st.session_state["X"] , feat_used)
        variance_df = create_pca_before(filter_data(st.session_state["X"], feat_used))
        chart = create_pca_plots(variance_df)
        st.altair_chart(chart, use_container_width=True)
    if variance_df is not None:
        n_comp = st.slider(
            label="Number of Components", min_value=1, max_value=variance_df.shape[0]
        )

        if st.button(label="Update data with Principal Components"):
            st.session_state["X"] = pd.concat(
                [
                    st.session_state["X"].drop(columns=feat_used),
                    create_pca_after(
                        filter_data(st.session_state["X"], feat_used), n_comp
                    ),
                ],
                axis=1,
            )
            st.write(st.session_state["X"].head())

        if st.button(
            label="Remove Principal Components (and add back the original features)"
        ):
            st.session_state["X"] = create_x_df(cont_df, cat_df)
            st.write(st.session_state["X"].head())
