import pandas as pd
import ydata_profiling
import streamlit as st
from streamlit_pandas_profiling import st_profile_report

st.set_page_config(page_title="Homepage", layout="wide")

pr = st.session_state["df"].profile_report(minimal=True)

st_profile_report(pr)
