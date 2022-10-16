import streamlit as st
import requests
from io import StringIO

st.title("SchNet Frontend")

st.header("Upload a file for MD prediction")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    # write sring_data to file
    #open text file
    text_file = open("data.xyz", "w")
    text_file.write(string_data)
    text_file.close()
    st.write(string_data)
    
st.header("MD Predictor API")
# value = st.text_input('Any 1 model input for md prediction', 'data.xyz')
if st.checkbox("Run"):
    data = requests.get('http://0.0.0.0:8080/mdpredictor/'+'data.xyz').text
    st.download_button('Download File', data)  # Defaults to 'text/plain'