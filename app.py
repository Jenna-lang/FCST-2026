import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import numpy as np

# 1. Cấu hình trang
st.set_page_config(page_title="AI Supply Chain Advisor", layout="wide")
st.title("🤖 AI Strategic Advisor: History & 2026 Forecast")

# 2. Sidebar - Quản lý File
st.sidebar.header("Data Settings")
uploaded_file = st.sidebar.file_uploader("Tải lên file dự báo (Excel)", type=['xlsx'])

# Hàm load dữ liệu linh hoạt
def load_data(file):
    df = pd.read_excel(file)
    df.columns = [str(col).strip() for col in df.columns]
    # Tự động nhận diện cột ngày tháng
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'Month' in col]
    if date_cols:
        df['ds'] = pd.to_datetime(df[date_cols[0]])
    return df

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        
        # Cấu hình cột động từ Sidebar
        cust_col = st.sidebar.selectbox("Cột tên Khách hàng:", df.columns, index=0)
        cie_col = st.sidebar.selectbox("Cột mã màu/CIE:", df.columns, index=1)
        
        # Tạo 2 Tab chính
        tab_strategic, tab_data = st.tabs(["🚀 AI Strategy & Pareto", "📊 Data Explorer"])

        with tab_strategic:
            # --- LOGIC PARETO (80/20) ---
            cust_list = sorted(df[cust_col].unique().astype(str))
            selected_cust = st.selectbox("1. Chọn Khách hàng:", cust_list)

            if selected_cust:
                cust_df = df[df[cust_col].astype(str) == selected_cust].copy()
                
                # Tính Pareto dựa trên số lượng hoặc giá trị (M USD)
                val_col = 'M USD' if 'M USD' in df.columns else 'Actual'
                if val_col in df.columns:
                    sales = cust_df.groupby('Material name')[val_col].sum().sort_values(ascending=False).reset_index()
                    sales['Cum_Pct'] = sales[val_col].cumsum() / sales[val_col].sum()
                    top_prods = sales[sales['Cum_Pct'] <= 0.81]['Material name'].unique()
                    
                    st.subheader(f"🎯 Phân tích Pareto cho {selected_cust}")
                    st.info(f"Top 20% mã hàng (Nhóm A) chiếm 80% giá trị của khách hàng này.")
                    
                    selected_prod = st.selectbox(f"2. Chọn Mã hàng chiến lược:", top_prods)

                    if selected_prod:
                        prod_df = cust_df[cust_df['Material name'] == selected_prod].copy()
                        cie_options = sorted(prod_df[cie_col].unique().astype(str))
                        selected_cies = st.multiselect("3. Chọn CIE Color Codes:", cie_options, default=cie_options[:1])

                        if selected_cies:
                            fig = go.Figure()
                            ai_insights = []
                            
                            for cie in selected_cies:
                                cie_df = prod_df[prod_df[cie_col].astype(str) == cie].copy()
                                if 'ds' in cie_df.columns:
                                    cie_df['m'] = cie_df['ds'].dt.to_period('M').dt.to_timestamp()
                                    actual = cie_df.groupby('m').size().reset_index().rename(columns={'m':'ds', 0:'y'})
                                    
                                    if len(actual) >= 2:
                                        m = Prophet(yearly_seasonality=True).fit(actual)
                                        future = m.make_future_dataframe(periods=12, freq='MS')
                                        fcst = m.predict(future)
                                        
                                        # Vẽ biểu đồ
                                        fig.add_trace(go.Scatter(x=actual['ds'], y=actual['y'], name=f"Thực tế - {cie}"))
                                        f_only = fcst[fcst['ds'] > actual['ds'].max()]
                                        fig.add_trace(go.Scatter(x=f_only['ds'], y=f_only['yhat'], name=f"FCST 2026 - {cie}", line=dict(dash='dash')))
                                        
                                        # AI Insight
                                        avg_past = actual['y'].tail(6).mean()
                                        max_26 = f_only['yhat'].max()
                                        trend = "Tăng trưởng" if max_26 > avg_past else "Sụt giảm"
                                        ai_insights.append(f"**CIE {cie}**: Dự kiến {trend} trong năm 2026.")

                            st.divider()
                            st.subheader("💡 AI Advisor Insights")
                            for insight in ai_insights:
                                st.success(insight)
                            
                            st.plotly_chart(fig, use_container_width=True)

        with tab_data:
            st.subheader("Bảng dữ liệu chi tiết")
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"Lỗi hệ thống: {e}")
else:
    st.info("Chào Jenna! Hãy tải file 'AICheck.xlsx' để kích hoạt AI Advisor.")
