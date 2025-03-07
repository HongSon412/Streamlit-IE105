import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st
import pickle
import numpy as np

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
    page_title="Model Evaluation",
    page_icon="🧫",
    layout="wide",
)
st.write("# Model Evaluation 🧫")


# Tải mô hình đã lưu
with open(r'C:\Users\asus\OneDrive - Trường ĐH CNTT - University of Information Technology\VISUAL STUDIO CODE\PYTHON\IE105\Streamlit IE105\Web_page\Models\clf.pkl', 'rb') as file:
    clf = pickle.load(file)

# Tải vectorizer đã lưu
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

st.write("### Model Evaluation Metrics")

st.markdown("""
            - **Accuracy Score**: The proportion of correctly predicted instances out of the total instances.
            - **Confusion Matrix**: A table that summarizes the number of correct and incorrect predictions for each class.
            - **Classification Report**: A detailed report showing precision, recall, F1-score, and accuracy for each class.
            """)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)  

# Chuyển đổi classification report thành DataFrame
class_report_df = pd.DataFrame(class_report).transpose()

st.write("### Accuracy Score")

# Hiển thị accuracy
st.write(f"Here we use a total of 5 different model: LogisticRegression, DecisionTree, RandomForest, SupportVectorMachine, NeuralNetworking. The model with the highest accuracy score is Random Forest with {accuracy*100:.2f}%")

# Hiển thị classification report dưới dạng bảng
st.write("### Classification Report")
st.dataframe(class_report_df)  # Hiển thị bảng trong Streamlit

# Vẽ confusion matrix dưới dạng heatmap
st.write("### Confusion Matrix")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['weak', 'normal', 'strong'], yticklabels=['weak', 'normal', 'strong'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
st.pyplot(fig)