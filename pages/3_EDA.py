import streamlit as st
import pandas as pd
import altair as alt
from eda import (
    boxplot,
    stacked_bar,
    scatter,
    cor_matrix,
    missing_value_plot,
)

st.set_page_config(page_title="Visualization", layout="wide")
alt.data_transformers.disable_max_rows()
cat_list = ["object", "category"]
exp1 = st.expander(label="Association Figures")

# small_df = st.session_state["df"].sample(n=10_000)
small_df = pd.read_csv(
    "C:\Projects\Rajk\prog_2\project\prog_machine_project\data\smoking_driking_dataset_Ver01.csv"
).sample(n=10_000)
with exp1:
    options = st.multiselect(
        label="Choose the desired colums",
        options=small_df.columns,
        default=small_df.columns[[1, 2]].to_list(),
        max_selections=2,
    )
    if len(options) == 2:
        st.write("Ok")
        # Both category:
        if (
            small_df[options[0]].dtype in cat_list
            and small_df[options[1]].dtype in cat_list
        ):
            fig = stacked_bar(small_df, options)
        # Both continous:
        elif (
            small_df[options[0]].dtype not in cat_list
            and small_df[options[1]].dtype not in cat_list
        ):
            fig = scatter(small_df, options)
        # One is continous, other is not:
        else:
            fig = boxplot(small_df, options)
        st.altair_chart(fig, use_container_width=True)
# st.write(len(small_df.select_dtypes(include=['int64', 'float64']).columns))
exp2 = st.expander(label="Correlation Matrix")

with exp2:
    fig = cor_matrix(small_df)
    st.altair_chart(fig, use_container_width=True)

exp3 = st.expander(label="Missing Values")
with exp3:
    fig = missing_value_plot(small_df)
    st.pyplot(fig)
