import streamlit as st
import pandas as pd
import modules_for_pages.data_wrangling as dw

df_x = dw.load_data()


feat_used = st.multiselect(
    "Choose the features to be included in PCA",
    options=df_x.columns,
    default=df_x.columns.to_list(),
)

df_for_dimred = dw.filter_data(df_x, feat_used)
variance_df = dw.create_pca_before(df_for_dimred)
chart = dw.create_plots(variance_df)
st.altair_chart(chart,use_container_width=True)

n_comp = st.slider(label="Number of Components",min_value=1,max_value=variance_df.shape[0])
pca_df = dw.create_pca_after(df_for_dimred,n_comp)
st.write(pca_df.head())

if st.button(label="Create data"):
    df_out = pd.concat([df_x.drop(columns=feat_used),pca_df],axis=1)
    st.write(df_out.head())