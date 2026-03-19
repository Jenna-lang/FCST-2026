import streamlit as st
import pandas as pd

st.title("Phân tích dự báo Lighting LED")

# Tạo khu vực tải file
uploaded_file = st.sidebar.file_uploader("Tải lên file Excel dự báo", type=['xlsx'])

if uploaded_file is not None:
    # Nếu bạn đã có file, app sẽ đọc file đó
    df = pd.read_excel(uploaded_file)
    st.success("Đã tải dữ liệu thành công!")
else:
    # Nếu chưa có file, app sẽ dùng dữ liệu tạm thời để không bị lỗi giao diện
    st.warning("Chưa có dữ liệu. Đang hiển thị dữ liệu mẫu để bạn hình dung:")
    data = {
        'Month': ['01/2026', '02/2026', '03/2026'],
        'Actual': [1000, 1200, 1100],
        'Forecast': [950, 1150, 1250]
    }
    df = pd.DataFrame(data)

# Sau đó mới chạy các đoạn code tính MAPE và vẽ biểu đồ với 'df' này
