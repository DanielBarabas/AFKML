import streamlit as st
import altair as alt
from modules.eda import (
    v_counts_bar_chart,
    boxplot,
    stacked_bar,
    scatter,
    cor_matrix,
    missing_value_plot,
)

st.set_page_config(page_title="Exploratory data analysis", layout="wide")
st.title("Exploratory data analysis")
alt.data_transformers.disable_max_rows()

dtype_map_inverse = {
    "float64": "Numeric",
    "category": "Categorical",
    "object": "Categorical",
    "int64": "Numeric",
}


# TODO add differing "key" param to buttons -> so that they can have the same name
with st.expander(label="Descriptive Statistics"):
    if st.button("Create descriptive statistics"):
        st.write(st.session_state["df"].describe())

# TODO not just object but category dtype as well!
with st.expander(label="Value counts for categorical data"):
    cat_options = (
        st.session_state["df"].select_dtypes(include="object").columns.tolist()
    )
    selected_cat = st.selectbox("Select categorical variable", cat_options)
    if st.button("Create value counts chart"):
        v_counts = st.session_state["df"].value_counts(selected_cat)

        fig = v_counts_bar_chart(v_counts)
        st.altair_chart(fig, use_container_width=True)


with st.expander(label="Association Figures"):
    options = st.multiselect(
        label="Choose the desired colums",
        options=st.session_state["df"].columns,
        default=st.session_state["df"].columns[[1, 2]].to_list(),
        max_selections=2,
    )
    st.write(str(st.session_state["df"][options[0]].dtype),str(st.session_state["df"][options[1]].dtype))
    if len(options) == 2 and st.button("Create association figure"):
        # Both category:
        if (
            dtype_map_inverse[str(st.session_state["df"][options[0]].dtype)] == dtype_map_inverse[str(st.session_state["df"][options[1]].dtype)] == "Categorical"

        ):
            st.write("Both Cat")
            fig = stacked_bar(st.session_state["df"], options)
        # Both continous:
        elif (
            dtype_map_inverse[str(st.session_state["df"][options[0]].dtype)] == dtype_map_inverse[str(st.session_state["df"][options[1]].dtype)] == "Numeric"
        ):
            st.write("Both Cont")
            fig = scatter(st.session_state["df"], options)
        # One is continous, other is not:
        else:
            st.write("Mixed")
            fig = boxplot(st.session_state["df"], options)
        st.altair_chart(fig, use_container_width=True)


with st.expander(label="Correlation Matrix"):
    if st.button("Create correlation matrix"):
        fig = cor_matrix(st.session_state["df"])
        st.altair_chart(fig, use_container_width=True)


with st.expander(label="Missing Values"):
    if st.button("Create missing values chart"):
        fig = missing_value_plot(st.session_state["df"])
        st.pyplot(fig)
