import ydata_profiling
import streamlit as st
from streamlit_ydata_profiling import st_profile_report

st.set_page_config(page_title="Pandas profiling", layout="wide")

# Don't run until no data is uploaded
if "df" not in st.session_state:
    st.write("First upload some data on the homepage")
    st.stop()


st.title("Pandas profiling")

pr = st.session_state["df"].profile_report(minimal=True)
st.write("ok")
# st_profile_report(pr)
