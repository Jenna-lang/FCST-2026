import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="FCST-2026 Lighting LED", layout="wide")

# --- 1. SIDEBAR: TẢI FILE ---
st.sidebar.header("Cài đặt dữ liệu")
uploaded_file = st.sidebar.file_uploader("Tải lên file Excel dự báo", type=["xlsx"])

if uploaded_file is not None:
    # Đọc dữ liệu
    df = pd.read_excel(uploaded_file)
    st.sidebar.success("Tải dữ liệu thành công!")

    # --- 2. GIAO DIỆN CHÍNH ---
    st.title("🚀 Hệ thống Dự báo Lighting LED 2026")

    # Tạo Tab để phân chia hiển thị
    tab_tong_quan, tab_chi_tiet = st.tabs(["📊 Tổng quan dự báo", "🎯 Chi tiết Khách hàng & Mã hàng"])

    with tab_tong_quan:
        st.subheader("Dữ liệu tổng hợp toàn bộ đơn hàng")
        # Tính MAPE tổng quát nếu có cột Actual và Forecast
        if 'Actual' in df.columns and 'Forecast' in df.columns:
            mask = df['Actual'] != 0
            mape = np.mean(np.abs((df['Actual'][mask] - df['Forecast'][mask]) / df['Actual'][mask])) * 100
            st.metric("Chỉ số MAPE (Toàn hệ thống)", f"{mape:.2f}%")
        
        st.dataframe(df, use_container_width=True)

    with tab_chi_tiet:
        st.subheader("Bộ lọc chi tiết")
        
        # Tự động nhận diện cột (Tránh lỗi KeyError)
        cust_col = 'Customer name' if 'Customer name' in df.columns else df.columns[1]
        mat_col = 'Material name' if 'Material name' in df.columns else df.columns[2]

        col1, col2 = st.columns(2)
        with col1:
            # Thêm key để tránh lỗi Duplicate ID
            selected_cust = st.multiselect("Chọn khách hàng", df[cust_col].unique(), key="cust_filter")
        
        with col2:
            if selected_cust:
                available_mats = df[df[cust_col].isin(selected_cust)][mat_col].unique()
                selected_mats = st.multiselect("Chọn mã hàng", available_mats, key="mat_filter")
            else:
                selected_mats = []

        # Hiển thị kết quả lọc
        if selected_cust and selected_mats:
            final_df = df[(df[cust_col].isin(selected_cust)) & (df[mat_col].isin(selected_mats))]
            
            st.write(f"Đang hiển thị {len(final_df)} bản ghi:")
            st.dataframe(final_df, use_container_width=True)
            
            # Vẽ biểu đồ so sánh nếu có cột Actual/Forecast
            if 'Actual' in final_df.columns and 'Forecast' in final_df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=final_df.index, y=final_df['Actual'], name='Thực tế'))
                fig.add_trace(go.Scatter(x=final_df.index, y=final_df['Forecast'], name='Dự báo', line=dict(dash='dash')))
                fig.update_layout(title="Xu hướng Thực tế vs Dự báo")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Vui lòng chọn Khách hàng và Mã hàng ở trên để xem chi tiết.")

else:
    st.info("👋 Chào Jenna! Hãy tải file Excel của bạn ở thanh bên trái để bắt đầu phân tích.")
