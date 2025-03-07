import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

#Importing necesssary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

# Ignore warnings
import warnings 
from warnings import filterwarnings
filterwarnings("ignore")

#Importing SHAP(XAI)
from streamlit_shap import st_shap
import shap
shap.initjs()

st.set_page_config(
    page_title="Data",
    page_icon="⚗️",
    layout="wide",
)
st.write("# Data ⚗️")


conn = sqlite3.connect(r'C:\Users\asus\OneDrive - Trường ĐH CNTT - University of Information Technology\VISUAL STUDIO CODE\PYTHON\IE105\Streamlit IE105\Datasets\password_data.sqlite')
data = pd.read_sql_query("SELECT * FROM Users" ,conn)

# Giao diện Streamlit
st.title("1. Data Collection")

# Thêm nút để xem dữ liệu
if st.button("Load and View Data"):
    # Kết nối và truy vấn dữ liệu
    st.write("### Data from Database:")
    st.dataframe(data)
else:
    st.write("Click the button above to load and view data.")

st.title("2. Data Cleaning")
data = data.drop('index', axis = 1)
# Thêm nút để xem dữ liệu
if st.button("Load and View Data Cleaning"):
    # Kết nối và truy vấn dữ liệu
    st.write("### Data from Database:")
    st.dataframe(data)
else:
    st.write("Click the button above to load and view data.")
    
    