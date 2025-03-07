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
    page_title="Data Analysis",
    page_icon="🧪",
    layout="wide",
)
st.write("# Data Analysis 🧪")


conn = sqlite3.connect(r'C:\Users\asus\OneDrive - Trường ĐH CNTT - University of Information Technology\VISUAL STUDIO CODE\PYTHON\IE105\Streamlit IE105\Datasets\password_data.sqlite')
data = pd.read_sql_query("SELECT * FROM Users" ,conn)

data["length"] = data["password"].str.len()
data = data.drop('index', axis = 1)

def freq_lowercase(row):
    return len([char for char in row if char.islower()])/len(row)

def freq_uppercase(row):
    return len([char for char in row if char.isupper()])/len(row)

def freq_numerical_case(row):
    return len([char for char in row if char.isdigit()])/len(row)

data["lowercase_freq"] = np.round(data["password"].apply(freq_lowercase) , 3)

data["uppercase_freq"] = np.round(data["password"].apply(freq_uppercase) , 3)

data["digit_freq"] = np.round(data["password"].apply(freq_numerical_case) , 3)

def freq_special_case(row):
    special_chars = [] 
    for char in row:
        if not char.isalpha() and not char.isdigit(): 
            special_chars.append(char) 
    return len(special_chars)

data["special_char_freq"] = np.round(data["password"].apply(freq_special_case) , 3)
data["special_char_freq"] = data["special_char_freq"]/data["length"]

if st.button("Load and View Feature Engineering"):
    # Kết nối và truy vấn dữ liệu
    st.write("### Data from Database:")
    st.dataframe(data)
else:
    st.write("Click the button above to load and view data.")

st.dataframe(data[["length" , "strength"]].groupby("strength").agg(["min", "max" , "mean" , "median"]))
st.dataframe(data[["lowercase_freq" , "strength"]].groupby("strength").agg(["min", "max" , "mean" , "median"]))
st.dataframe(data[["uppercase_freq" , "strength"]].groupby("strength").agg(["min", "max" , "mean" , "median"]))
st.dataframe(data[["digit_freq" , "strength"]].groupby("strength").agg(["min", "max" , "mean" , "median"]))
st.dataframe(data[["special_char_freq" , "strength"]].groupby("strength").agg(["min", "max" , "mean" , "median"]))

# Tạo lưới các biểu đồ
fig, axes = plt.subplots(3, 2, figsize=(18, 12))  # Tăng kích thước biểu đồ

# Danh sách các cột và tiêu đề
columns = [
    ("length", "Length by Strength"),
    ("lowercase_freq", "Lowercase Frequency by Strength"),
    ("uppercase_freq", "Uppercase Frequency by Strength"),
    ("digit_freq", "Digit Frequency by Strength"),
    ("special_char_freq", "Special Character Frequency by Strength"),
]

# Vẽ các biểu đồ hộp
for ax, (column, title) in zip(axes.flat[:-1], columns):  # Bỏ ô cuối nếu trống
    sns.boxplot(
        x="strength",
        y=column,
        hue="strength",
        data=data,
        ax=ax,
        palette="Set2",  # Thay đổi màu sắc
        showmeans=True,  # Hiển thị trung bình
        linewidth=1.2,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Strength")
    ax.set_ylabel(column.capitalize().replace("_", " "))
    ax.legend_.remove()  # Xóa chú giải trùng lặp

# Chỉ giữ lại chú giải ở ô đầu tiên
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, title="Strength")

# Xóa trục của ô trống cuối cùng (nếu có)
axes.flat[-1].axis("off")

# Tăng khoảng cách giữa các biểu đồ
plt.subplots_adjust(hspace=0.5, wspace=0.4)

# Hiển thị biểu đồ trong Streamlit
st.pyplot(fig)

# Hàm hiển thị phân phối dữ liệu với Streamlit
def get_dist(data, feature):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # Tăng kích thước để đẹp hơn
    
    # Biểu đồ violin
    sns.violinplot(x='strength', y=feature, data=data, ax=axes[0])
    axes[0].set_title(f"Violin Plot for {feature} by Strength")
    axes[0].set_xlabel("Strength")
    axes[0].set_ylabel(feature)
    
    # Biểu đồ phân phối
    sns.kdeplot(data[data['strength'] == 0][feature], color="red", label="0", ax=axes[1], fill=True)
    sns.kdeplot(data[data['strength'] == 1][feature], color="blue", label="1", ax=axes[1], fill=True)
    sns.kdeplot(data[data['strength'] == 2][feature], color="orange", label="2", ax=axes[1], fill=True)
    axes[1].set_title(f"Distribution Plot for {feature} by Strength")
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel("Density")
    axes[1].legend(title="Strength")

    # Hiển thị trong Streamlit
    st.pyplot(fig)

# Ứng dụng Streamlit
st.title("Feature Distribution Analysis")
st.write("Use the dropdown and button below to analyze the selected feature.")

# Dropdown để chọn cột từ dữ liệu
feature = st.selectbox("Select a feature to analyze:", data.columns[2:])

# Nút bấm
if st.button("Generate Plot"):  # Khi nhấn nút, biểu đồ sẽ được tạo
    if feature:
        get_dist(data, feature)
    else:
        st.warning("Please select a feature to analyze.")