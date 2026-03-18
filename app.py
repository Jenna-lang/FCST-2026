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
    # Xóa khoảng trắng thừa ở tên cột
    df.columns = [str(col).strip() for col in df.columns]
    df['ds'] = pd.to_datetime(df['Requested deliv. date'])
    return df

try:
    df = load_data()
    
    # --- PHẦN CẤU HÌNH SIDEBAR ---
    st.sidebar.header("⚙️ Cấu hình hiển thị")
    
    # Cho phép người dùng tự chọn cột nào là "Tên Khách Hàng"
    # Bạn chỉ cần chọn cột có tên như 'End Cust Name' hoặc 'Customer' ở đây
    cust_col = st.sidebar.selectbox(
        "Chọn cột hiển thị Tên Khách Hàng:", 
        options=df.columns,
        index=0
    )

    # Lọc danh sách sản phẩm Pareto (85% giá trị)
    summary = df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
    summary['Cum_Pct'] = summary['M USD'].cumsum() / summary['M USD'].sum()
    pareto_list = summary[summary['Cum_Pct'] <= 0.85]['Material name'].unique()

    st.sidebar.divider()
    selected_prod = st.sidebar.selectbox("1. Chọn mã linh kiện Pareto:", pareto_list)

    if selected_prod:
        prod_df = df[df['Material name'] == selected_prod].copy()
        tab1, tab2 = st.tabs(["📊 Tổng quan mã hàng", "🎯 Dự báo & Lời khuyên AI"])

        with tab1:
            st.subheader(f"Phân tích tỷ trọng khách hàng cho: {selected_prod}")
            c_a, c_b = st.columns(2)
            with c_a:
                # Biểu đồ tròn theo Tên khách hàng đã chọn
                fig_pie = px.pie(prod_df.groupby(cust_col)['M USD'].sum().reset_index(), 
                                 values='M USD', names=cust_col, hole=0.4, title="Tỷ trọng doanh số")
                st.plotly_chart(fig_pie, use_container_width=True)
            with c_b:
                prod_df['Month'] = prod_df['ds'].dt.to_period('M').dt.to_timestamp()
                fig_bar = px.bar(prod_df.groupby(['Month', cust_col])['Order qty.(A)'].sum().reset_index(), 
                                 x='Month', y='Order qty.(A)', color=cust_col, title="Lịch sử đặt hàng")
                st.plotly_chart(fig_bar, use_container_width=True)

        with tab2:
            customers = sorted([str(x) for x in prod_df[cust_col].unique()])
            selected_cust = st.selectbox(f"Chọn khách hàng cụ thể:", customers)
            
            if selected_cust:
                cust_data = prod_df[prod_df[cust_col] == selected_cust].copy()
                cust_data['ds_month'] = cust_data['ds'].dt.to_period
