import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import numpy as np

# 1. Cấu hình trang
st.set_page_config(page_title="AI Supply Chain Advisor", layout="wide")
st.title("🤖 AI Strategic Advisor: Actual vs Forecast 2026")

# 2. Sidebar - Tải dữ liệu
st.sidebar.header("Data Settings")
uploaded_file = st.sidebar.file_uploader("Tải lên file dự báo (Excel)", type=['xlsx'])

if uploaded_file is not None:
    try:
        # Load dữ liệu và chuẩn hóa tên cột
        df = pd.read_excel(uploaded_file)
        df.columns = [str(col).strip() for col in df.columns]
        
        # Nhận diện cột ngày tháng
        for col in df.columns:
            if 'date' in col.lower() or 'month' in col.lower() or 'ds' in col.lower():
                df['ds'] = pd.to_datetime(df[col], errors='coerce')
                break

        # 3. Phân chia Tab
        tab_fcst, tab_compare, tab_raw = st.tabs(["🚀 Dự báo AI (Prophet)", "📊 So sánh Actual vs FCST", "📋 Dữ liệu gốc"])

        with tab_fcst:
            st.subheader("Dự báo xu hướng 2026 cho Mã hàng chiến lược")
            # Logic Pareto để chọn mã hàng quan trọng
            cust_col = 'Customer name' if 'Customer name' in df.columns else df.columns[0]
            selected_cust = st.selectbox("Chọn Khách hàng:", df[cust_col].unique(), key="fcst_cust")
            
            cust_df = df[df[cust_col] == selected_cust]
            # Giả định dùng cột 'Actual' hoặc 'M USD' để tính Pareto
            val_col = 'Actual' if 'Actual' in df.columns else df.select_dtypes(include=[np.number]).columns[0]
            
            sales = cust_df.groupby('Material name')[val_col].sum().sort_values(ascending=False).reset_index()
            sales['Cum_Pct'] = sales[val_col].cumsum() / sales[val_col].sum()
            top_prods = sales[sales['Cum_Pct'] <= 0.85]['Material name'].unique()
            
            selected_prod = st.selectbox("Chọn Mã hàng (Nhóm Pareto A):", top_prods, key="fcst_prod")
            
            if selected_prod:
                prod_df = cust_df[cust_df['Material name'] == selected_prod].copy()
                # Chạy Prophet
                actual_data = prod_df.groupby('ds')[val_col].sum().reset_index()
                actual_data.columns = ['ds', 'y']
                
                if len(actual_data) >= 2:
                    m = Prophet(yearly_seasonality=True).fit(actual_data)
                    future = m.make_future_dataframe(periods=12, freq='MS')
                    fcst = m.predict(future)
                    
                    fig_prophet = go.Figure()
                    fig_prophet.add_trace(go.Scatter(x=actual_data['ds'], y=actual_data['y'], name="Lịch sử thực tế"))
                    fig_prophet.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat'], name="Dự báo AI", line=dict(dash='dash')))
                    st.plotly_chart(fig_prophet, use_container_width=True)
                    
                    # AI Advisor
                    st.info(f"💡 **AI Advisor:** Mã hàng {selected_prod} có xu hướng ổn định dựa trên dữ liệu lịch sử.")

        with tab_compare:
            st.subheader("Phân tích sai số: Thực tế (Actual) vs Kế hoạch (FCST)")
            if 'Actual' in df.columns and 'Forecast' in df.columns:
                # Tính MAPE
                mask = (df['Actual'] > 0) & (df['Forecast'] > 0)
                temp_df = df[mask].copy()
                mape = np.mean(np.abs((temp_df['Actual'] - temp_df['Forecast']) / temp_df['Actual'])) * 100
                
                col1, col2 = st.columns(2)
                col1.metric("Độ chính xác dự báo", f"{100-mape:.2f}%")
                col2.metric("Sai số MAPE", f"{mape:.2f}%")
                
                # Biểu đồ cột so sánh
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(x=df['ds'], y=df['Actual'], name="Actual (Thực tế)"))
                fig_comp.add_trace(go.Bar(x=df['ds'], y=df['Forecast'], name="FCST (Kế hoạch)"))
                fig_comp.update_layout(barmode='group', title="So sánh Actual vs Forecast theo thời gian")
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.warning("File Excel cần có cột 'Actual' và 'Forecast' để hiển thị phần này.")

        with tab_raw:
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"Lỗi: {e}")
else:
    st.info("Vui lòng tải file Excel để khôi phục các phân tích.")
