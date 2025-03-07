import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide",
)
#Importing necesssary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore warnings
import warnings 
from warnings import filterwarnings
filterwarnings("ignore")

#Importing SHAP(XAI)
from streamlit_shap import st_shap
import shap
shap.initjs()