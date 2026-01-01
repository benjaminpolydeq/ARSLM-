import streamlit as st

st.set_page_config(page_title="ARSLM", layout="centered")

st.title("🤖 ARSLM")
st.write("ARSLM is running correctly on Streamlit 🚀")

prompt = st.text_input("Ask something")

if prompt:
    st.success(f"You asked: {prompt}")
