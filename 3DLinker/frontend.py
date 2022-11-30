import streamlit as st
import requests

st.title("3DLinker Frontend")
st.header("GenerationAPI")
value = st.text_input('Any 1 dataset input for generation', 'qm9/zinc/cep')
if value in ["zinc","qm9","cep"]:
    st.write("Works")
    data = requests.get('http://0.0.0.0:8080/generate/'+value).text
    st.download_button('Download File', data)
else:
    st.write("Doesn't work, Check Input")
    
st.header("EvaluationAPI")
if st.checkbox("Run"):
    data =requests.get('http://0.0.0.0:8080/evaluation/1').text
    st.download_button('Download File', data)  # Defaults to 'text/plain'