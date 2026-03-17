import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Hệ thống Dự báo Pareto 2026", layout="wide")
st.title("🚀 Hệ Thống Phân Tích Cung Ứng Pareto 2026")

@st.cache_data
def load_data():
    df = pd.read_excel('AICheck.xlsx')
    # Chuẩn hóa tên cột: xóa khoảng trắng dư thừa
    df.columns = [str(col).strip() for col in df.columns]
    df['ds'] = pd.to_datetime(df['Requested deliv. date'])
    return df

try:
    df = load_data()
    
    # Tự động nhận diện cột Tên khách hàng (Thường là 'End Cust' hoặc 'Customer Name')
    # Ưu tiên tìm cột có chữ 'Cust' và không phải là 'Code'
    cust_col = next((col for col in df.columns if 'cust' in col.lower() and 'code' not in col.lower()), None)
    
    if not cust_col:
        # Nếu không tìm thấy, lấy cột 'End Cust' mặc định
        cust_col = 'End Cust' if 'End Cust' in df.columns else df.columns[0]

    # Lọc danh sách sản phẩm Pareto (85% giá trị)
    summary = df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
    summary['Cum_Pct'] = summary['M USD'].cumsum() / summary['M USD'].sum()
    pareto_list = summary[summary['Cum_Pct'] <= 0.85]['Material name'].unique()

    st.sidebar.header("🔍 Bộ Lọc Truy Vấn")
    selected_prod = st.sidebar.selectbox("1. Chọn mã linh kiện Pareto:", pareto_list)

    if selected_prod:
        prod_df = df[df['Material name'] == selected_prod].copy()
        
        tab1, tab2 = st.tabs(["📊 Tổng quan mã hàng", "🎯 Dự báo & Lời khuyên AI theo Khách hàng"])

        with tab1:
            st.subheader(f"Phân tích tỷ trọng khách hàng cho: {selected_prod}")
            col_a, col_b = st.columns(2)
            with col_a:
                cust_share = prod_df.groupby(cust_col)['M USD'].sum().reset_index()
                fig_pie = px.pie(cust_share, values='M USD', names=cust_col, hole=0.4, title="Tỷ trọng doanh số (M USD)")
                st.plotly_chart(fig_pie, use_container_width=True)
            with col_b:
                prod_df['Month'] = prod_df['ds'].dt.to_period('M').dt.to_timestamp()
                monthly_cust = prod_df.groupby(['Month', cust_col])['Order qty.(A)'].sum().reset_index()
                fig_bar = px.bar(monthly_cust, x='Month', y='Order qty.(A)', color=cust_col, title="Lịch sử đặt hàng theo tháng")
                st.plotly_chart(fig_bar, use_container_width=True)

        with tab2:
            customers = sorted(prod_df[cust_col].unique())
            selected_cust = st.selectbox("Chọn Tên Khách Hàng cần xem chi tiết:", customers)
            
            if selected_cust:
                cust_data = prod_df[prod_df[cust_col] == selected_cust].copy()
                cust_data['ds_month'] = cust_data['ds'].dt.to_period('M').dt.to_timestamp()
                p_df = cust_data.groupby('ds_month')['Order qty.(A
