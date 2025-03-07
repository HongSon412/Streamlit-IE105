import streamlit as st
import pickle
import numpy as np
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

password = Image.open(r"C:\Users\asus\OneDrive - Trường ĐH CNTT - University of Information Technology\VISUAL STUDIO CODE\PYTHON\IE105\Streamlit IE105\Web_page\Images\password.jpg")
st.image(password, width=100)
st.write("# Predict Password Strength 🔒")

# Tải mô hình đã lưu
with open(r'C:\Users\asus\OneDrive - Trường ĐH CNTT - University of Information Technology\VISUAL STUDIO CODE\PYTHON\IE105\Streamlit IE105\Web_page\Models\rf.pkl', 'rb') as file:
    rf = pickle.load(file)

# Tải mô hình đã lưu
with open(r'C:\Users\asus\OneDrive - Trường ĐH CNTT - University of Information Technology\VISUAL STUDIO CODE\PYTHON\IE105\Streamlit IE105\Web_page\Models\vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

with open(r'C:\Users\asus\OneDrive - Trường ĐH CNTT - University of Information Technology\VISUAL STUDIO CODE\PYTHON\IE105\Streamlit IE105\Web_page\Variable\X_test.pkl', 'rb') as file:
    X_test = pickle.load(file)
with open(r'C:\Users\asus\OneDrive - Trường ĐH CNTT - University of Information Technology\VISUAL STUDIO CODE\PYTHON\IE105\Streamlit IE105\Web_page\Variable\X_train.pkl', 'rb') as file:
    X_train = pickle.load(file)
with open(r'C:\Users\asus\OneDrive - Trường ĐH CNTT - University of Information Technology\VISUAL STUDIO CODE\PYTHON\IE105\Streamlit IE105\Web_page\Variable\y_train.pkl', 'rb') as file:
    y_train = pickle.load(file)
with open(r'C:\Users\asus\OneDrive - Trường ĐH CNTT - University of Information Technology\VISUAL STUDIO CODE\PYTHON\IE105\Streamlit IE105\Web_page\Variable\y_test.pkl', 'rb') as file:
    y_test = pickle.load(file)
with open(r'C:\Users\asus\OneDrive - Trường ĐH CNTT - University of Information Technology\VISUAL STUDIO CODE\PYTHON\IE105\Streamlit IE105\Web_page\Variable\y_pred.pkl', 'rb') as file:
    y_pred = pickle.load(file)
with open(r'C:\Users\asus\OneDrive - Trường ĐH CNTT - University of Information Technology\VISUAL STUDIO CODE\PYTHON\IE105\Streamlit IE105\Web_page\Variable\explainer.pkl', 'rb') as file:
    explainer = pickle.load(file)
with open(r'C:\Users\asus\OneDrive - Trường ĐH CNTT - University of Information Technology\VISUAL STUDIO CODE\PYTHON\IE105\Streamlit IE105\Web_page\Variable\shap_values.pkl', 'rb') as file:
    shap_values = pickle.load(file)
with open(r'C:\Users\asus\OneDrive - Trường ĐH CNTT - University of Information Technology\VISUAL STUDIO CODE\PYTHON\IE105\Streamlit IE105\Web_page\Variable\feature_names.pkl', 'rb') as file:
    feature_names = pickle.load(file)

#Hàm dự đoán mật khẩu
def predict(password):
    # Biến đổi mật khẩu thành ma trận đặc trưng
    sample_array = np.array([password])
    sample_matrix = vectorizer.transform(sample_array) 
    
    # Tính toán các đặc trưng bổ sung
    length_pass = len(password)
    length_normalised_lowercase = len([char for char in password if char.islower()]) / len(password)
    
    new_matrix2 = np.append(sample_matrix.toarray(), (length_pass, length_normalised_lowercase)).reshape(1, 101)
    
    # Dự đoán kết quả
    result = rf.predict(new_matrix2)
    predicted_strength = ["weak", "normal", "strong"][result[0]]
    
    # Tính toán giá trị SHAP để giải thích
    shap_values = explainer.shap_values(new_matrix2)
    #explanation = shap.force_plot(explainer.expected_value[result[0]], shap_values[:,:,result[0]][0], feature_names, matplotlib=True)
    
    # Hiển thị kết quả và giải thích
    st.write(f"Password strength prediction: {predicted_strength}")
    st.write("Explanation of the prediction:")
    #st_shap(explanation)
    
    #return predicted_strength
     # Giải thích bằng force plot
    st.write("Force Plot (Feature Impact)")
    force_plot = shap.force_plot(
        explainer.expected_value[result[0]], 
        shap_values[:,:,result[0]][0], 
        feature_names, 
        matplotlib=True
    )
    st_shap(force_plot)
    
    # Giải thích bằng summary plot
    st.write("Summary Plot (Global Feature Importance)")
    summary_plot = shap.summary_plot(
        shap_values[:,:,result[0]], 
        new_matrix2, 
        feature_names=feature_names,
        plot_type="bar"
    )
    st_shap(summary_plot)

# Nhập dữ liệu từ người dùng
text = st.text_input("Enter a password:")

# Thêm nút để thực hiện dự đoán
if st.button("Predict Password Strength"):
    if text:  # Kiểm tra nếu người dùng đã nhập dữ liệu
        predict(text)
    else:
        st.write("Please input a value.")

st.sidebar.success("Select for more details.")

st.markdown(
    """
    - Strength of password is a vital task in understanding the security level of passwords. 
    
    - In this project, we employ SHAP to explain the model we used to analysis password strength based on their sentiment. 
    - The objective is to build accurate models that can automatically predict whether password is weak, normal or strong.

    **👈 Select for more details from the sidebar** to see some examples
    ### About our dataset
    - Check out [Kaggle 💾](https://www.kaggle.com/datasets/soylevbeytullah/password-datas)

    - This dataset having 100k passwords for natural language processing or Text analytics.
    - This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. 
    - We provide a set of 80,000 datasets for training and 20,000 for testing. 
    - So, predict by using LogisticRegression, DecisionTree, RandomForest, SupportVectorMachine, NeuralNetworking.
    - The best model is RandomForest with 0.92 accuracy.
"""
)