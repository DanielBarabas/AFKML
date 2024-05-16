import pandas as pd
import ydata_profiling
import streamlit as st
from streamlit_pandas_profiling import st_profile_report

st.set_page_config(page_title="Homepage", page_icon="🐼")

df = pd.read_csv("C:\Projects\Rajk\prog_2\project\prog_machine_project\data\smoking_driking_dataset_Ver01.csv")

pr = df.profile_report(minimal = True)

st_profile_report(pr)
