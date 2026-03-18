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
    df.columns = [str(col).strip() for col in df.columns]
    df['ds'] = pd.to_datetime(df['Requested deliv. date'])
    return df

try:
    df = load_data()
    
    # --- CẤU HÌNH SIDEBAR ---
    st.sidebar.header("⚙️ Cấu hình hiển thị")
    # Bước này giúp bạn chọn đúng cột 'Tên khách hàng' thay vì 'Mã khách hàng'
    cust_col = st.sidebar.selectbox("Chọn cột Tên Khách Hàng:", df.columns, 
                                     index=list(df.columns).index('End Cust') if 'End Cust' in df.columns else 0)

    # Lọc Pareto (85%)
    summary = df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
    summary['Cum_Pct'] = summary['M USD'].cumsum() / summary['M USD'].sum()
    pareto_list = summary[summary['Cum_Pct'] <= 0.85]['Material name'].unique()

    st.sidebar.divider()
    selected_prod = st.sidebar.selectbox("1. Chọn mã linh kiện Pareto:", pareto_list)

    if selected_prod:
        prod_df = df[df['Material name'] == selected_prod].copy()
        tab1, tab2 = st.tabs(["📊 Tổng quan mã hàng", "🎯 Dự báo & Lời khuyên AI 2026"])

        with tab1:
            st.subheader(f"Phân tích tỷ trọng cho: {selected_prod}")
            c_a, c_b = st.columns(2)
            with c_a:
                fig_pie = px.pie(prod_df.groupby(cust_col)['M USD'].sum().reset_index(), 
                                 values='M USD', names=cust_col, hole=0.4, title="Tỷ trọng doanh số theo khách hàng")
                st.plotly_chart(fig_pie, use_container_width=True)
            with c_b:
                prod_df['Month'] = prod_df['ds'].dt.to_period('M').dt.to_timestamp()
                fig_bar = px.bar(prod_df.groupby(['Month', cust_col])['Order qty.(A)'].sum().reset_index(), 
                                 x='Month', y='Order qty.(A)', color=cust_col, title="Lịch sử đặt hàng toàn bộ khách hàng")
                st.plotly_chart(fig_bar, use_container_width=True)

        with tab2:
            # --- THÊM LỰA CHỌN ALL ---
            cust_list = sorted([str(x) for x in prod_df[cust_col].unique()])
            options = ["ALL (Tổng hợp tất cả khách hàng)"] + cust_list
            selected_cust = st.selectbox("Chọn khách hàng cụ thể hoặc xem Tổng quát:", options)
            
            # Xử lý dữ liệu dựa trên lựa chọn
            if selected_cust == "ALL (Tổng hợp tất cả khách hàng)":
                cust_data = prod_df.copy()
                title_suffix = "TẤT CẢ KHÁCH HÀNG"
            else:
                cust_data = prod_df[prod_df[cust_col] == selected_cust].copy()
                title_suffix = selected_cust

            cust_data['ds_month'] = cust_data['ds'].dt.to_period('M').dt.to_timestamp()
            p_df = cust_data.groupby('ds_month')['Order qty.(A)'].sum().reset_index().rename(columns={'ds_month':'ds', 'Order qty.(A)':'y'})

            if len(p_df) >= 2:
                m = Prophet(yearly_seasonality=True).fit(p_df)
                future = m.make_future_dataframe(periods=12, freq='MS')
                forecast = m.predict(future)
                # Chặn giá trị âm cho
