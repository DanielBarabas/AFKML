import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Visualization", page_icon="📈")
alt.data_transformers.disable_max_rows()

exp =  st.expander(label="Association Figure")
df = pd.read_csv("C:\Projects\Rajk\prog_2\project\prog_machine_project\data\smoking_driking_dataset_Ver01.csv")
small_df = df.sample(n = 10_000)
with exp:
    st.write("asdfg")
    options = st.multiselect(label = "Choose the desired colums",
                             options=df.columns, 
                             default=df.columns[[1,2]].to_list(),
                             max_selections=2)
    if len(options) == 2:
        st.write("Ok")

        cc_fig = alt.Chart(small_df).mark_bar().encode(
            x=alt.X('count()').stack("normalize"),
            y=options[0],
            color=options[1]
        )
        st.altair_chart(cc_fig)
    #st.bar_chart({"data": [1, 5, 2, 6, 2, 1]})
