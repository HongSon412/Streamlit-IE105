import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import pandas as pd
st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide",
)

# Ignore warnings
import warnings 
from warnings import filterwarnings
filterwarnings("ignore")

common_password = Image.open(r"C:\Users\asus\OneDrive - Trường ĐH CNTT - University of Information Technology\VISUAL STUDIO CODE\PYTHON\IE105\Streamlit IE105\Web_page\Images\common_password.jpg")
st.image(common_password, width=1000)
st.write("# How Common is your Password? 🔒")

# Tạo dữ liệu giả lập
file_path = r"C:\Users\asus\OneDrive - Trường ĐH CNTT - University of Information Technology\VISUAL STUDIO CODE\PYTHON\IE105\Streamlit IE105\Datasets\top_10000_common_passwords.csv"
df = pd.read_csv(file_path, header=0, names=["password", "rank"])  # Đọc với tiêu đề đúng

# Chuyển đổi cột mật khẩu thành danh sách
password_list = df["password"].astype(str).str.strip().tolist()

# Hiển thị thông tin
st.write("Write a password to check how common it is. Let's see if your password is among the top 10,000 common passwords.")

def check_common(password):
    """Kiểm tra xem mật khẩu có trong danh sách 10,000 mật khẩu phổ biến không"""
    password = password.strip()  # Loại bỏ khoảng trắng
    if password in password_list:
        rank = df.loc[df["password"] == password, "rank"].values[0]  # Lấy thứ hạng
        st.write(f"""🔴 The password `{password}` is ranked **#{rank}** in the list of top 10,000 common passwords.""")
    else:
        st.write(f"✅ Your password `{password}` is **not common**.")


text = st.text_input("Enter a password:")

# Thêm nút để thực hiện dự đoán
if st.button("Check how common is your password"):
    # Hàm dự đoán mật khẩu
    if text:  # Kiểm tra nếu người dùng đã nhập dữ liệu
        check_common(text)
    else:
        st.write("Please input a value.")


# Hiển thị markdown trước
st.markdown(
    """
    - Do you know that the most common password is `123456`?
    
    - Don't believe me?, Check out [Wikipedia 📠](https://en.wikipedia.org/wiki/Wikipedia:10,000_most_common_passwords)
    - Let me help you to find out how common is your password.

    ### About our dataset
    - We crawl raw data from the internet and then clean it to get the final dataset.
    - Click the button below to download the dataset.
    """
)

# Tạo nút tải xuống
st.download_button(label="Click here to download ✨", 
                   data=file_path, 
                   file_name="common_passwords.txt", 
                   mime="text/plain")