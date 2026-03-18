import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Dự báo Pareto 2026", layout="wide")
st.title("🚀 Hệ Thống Phân Tích Cung Ứng Pareto 2026")

@st.cache_data
def load_data():
    df = pd.read_excel('AICheck.xlsx')
    # Xóa khoảng trắng thừa ở tên cột để tránh lỗi không tìm thấy cột
    df.columns = [str(col).strip() for col in df.columns]
    df['ds'] = pd.to_datetime(df['Requested deliv. date'])
    return df

try:
    df = load_data()
    
    # --- CẤU HÌNH SIDEBAR ---
    st.sidebar.header("⚙️ Cấu hình dữ liệu")
    # Cho phép bạn chọn đúng cột Tên Khách Hàng từ danh sách cột thực tế trong Excel
    # Bước này giúp sửa lỗi 'End Cust' nếu tên cột trong file bị thay đổi
    cust_col = st.sidebar.selectbox("Chọn cột Tên Khách Hàng:", df.columns)

    # Lọc danh sách Pareto (85% giá trị doanh số)
    summary = df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
    summary['Cum_Pct'] = summary['M USD'].cumsum() / summary['M USD'].sum()
    pareto_list = summary[summary['Cum_Pct'] <= 0.85]['Material name'].unique()

    st.sidebar.divider()
    selected_prod = st.sidebar.selectbox("Chọn mã linh kiện Pareto:", pareto_list)

    if selected_prod:
        prod_df = df[df['Material name'] == selected_prod].copy()
        tab1, tab2 = st.tabs(["📊 Tổng quan mã hàng", "🎯 Dự báo & Lời khuyên AI 2026"])

        with tab1:
            st.subheader(f"Phân tích tỷ trọng khách hàng cho: {selected_prod}")
            c1, c2 = st.columns(2)
            with c1:
                fig_pie = px.pie(prod_df.groupby(cust_col)['M USD'].sum().reset_index(), 
                                 values='M USD', names=cust_col, hole=0.4, title="Tỷ trọng theo doanh số")
                st.plotly_chart(fig_pie, use_container_width=True)
            with c2:
                prod_df['Month'] = prod_df['ds'].dt.to_period('M').dt.to_timestamp()
                fig_bar = px.bar(prod_df.groupby(['Month', cust_col])['Order qty.(A)'].sum().reset_index(), 
                                 x='Month', y='Order qty.(A)', color=cust_col, title="Biến động đơn đặt hàng")
                st.plotly_chart(fig_bar, use_container_width=True)

        with tab2:
            # --- THÊM LỰA CHỌN ALL ---
            cust_list = sorted([str(x) for x in prod_df[cust_col].unique()])
            options = ["ALL (Tổng hợp tất cả khách hàng)"] + cust_list
            selected_cust = st.selectbox("Chọn khách hàng cụ thể hoặc xem Tổng quát:", options)
            
            # Xử lý lọc dữ liệu
            if selected_cust == "ALL (Tổng hợp tất cả khách hàng)":
