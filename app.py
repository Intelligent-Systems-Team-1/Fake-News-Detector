import streamlit as st

st.title("Machine Learning Model App")
st.write("Upload your data and get predictions!")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    import pandas as pd
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)  # Show uploaded data

st.write("More features coming soon! 🚀")
