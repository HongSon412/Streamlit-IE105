import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image

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
data = r"C:\Users\asus\OneDrive - Trường ĐH CNTT - University of Information Technology\VISUAL STUDIO CODE\PYTHON\IE105\Streamlit IE105\Datasets\top_10000_common_passwords.csv"

st.write("Write a password to check how common it is. Let see if your password is among the top 10,000 common passwords.")

def def_common(password):
    # Đọc dữ liệu từ tệp
    with open(data, "r") as file:
        common_passwords = file.readlines()
    # Kiểm tra mật khẩu
    if password in common_passwords:
        st.write(f"""The password `{password}` is ranked {common_passwords.index(password)+1} in the list of top 10,000 common passwords.""")
        st.write("You should change your password.")
    else:
        st.write(f"Your password `{password}` is not common.")
        st.write("You are safe!")

text = st.text_input("Enter a password:")

# Thêm nút để thực hiện dự đoán
if st.button("Check how common is your password"):
    # Hàm dự đoán mật khẩu
    if text:  # Kiểm tra nếu người dùng đã nhập dữ liệu
        def_common(text)
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
                   data=data, 
                   file_name="common_passwords.txt", 
                   mime="text/plain")