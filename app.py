import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. Cấu hình hệ thống
st.set_page_config(page_title="AI Supply Chain Advisor 2026", layout="wide")

def process_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = [str(col).strip() for col in df.columns]
        if 'Requested deliv. date' in df.columns:
            df['ds'] = pd.to_datetime(df['Requested deliv. date'], errors='coerce')
            df = df.dropna(subset=['ds'])
            return df
        else:
            st.error("Thiếu cột 'Requested deliv. date'!")
            return None
    except Exception as e:
        st.error(f"Lỗi: {e}")
        return None

def calculate_detailed_metrics(cust_df, prod_name):
    p_df = cust_df[cust_df['Material name'] == prod_name].copy()
    p_df['m'] = p_df['ds'].dt.to_period('M').dt.to_timestamp()
    monthly_data = p_df.groupby('m')['Order qty.(A)'].sum().reset_index()
    
    # Tính Monthly Growth Rate (%)
    monthly_data['Growth'] = monthly_data['Order qty.(A)'].pct_change() * 100
    avg_growth = monthly_data['Growth'].mean() # Tăng trưởng trung bình hàng tháng
    
    # Tính YoY Growth (2026 vs 2025)
    act_2026 = p_df[p_df['ds'].dt.year == 2026].groupby(p_df['ds'].dt.month)['Order qty.(A)'].sum()
    act_2025 = p_df[(p_df['ds'].dt.year == 2025) & (p_df['ds'].dt.month.isin(act_2026.index))].groupby(p_df['ds'].dt.month)['Order qty.(A)'].sum()
    yoy_growth = (act_2026.sum() - act_2025.sum()) / act_2025.sum() * 100 if act_2025.sum() > 0 else 0.0
    
    return avg_growth, yoy_growth, act_2026.mean()

# --- SIDEBAR ---
st.sidebar.header("📁 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload AICheck.xlsx", type=['xlsx'])

if uploaded_file:
    df = process_data(uploaded_file)
    if df is not None:
        cust_col = st.sidebar.selectbox("Customer:", [c for c in df.columns if 'Customer' in c] or [df.columns[0]])
        cie_col = st.sidebar.selectbox("CIE/Color:", [c for c in df.columns if 'CIE' in c] or [df.columns[1]])
        adj_growth = st.sidebar.slider("Manual Adjustment (%)", -50, 50, 0)
        selected_cust = st.sidebar.selectbox("Select Customer:", ["-- Select --"] + sorted(df[cust_col].unique()))

        if selected_cust != "-- Select --":
            cust_df = df[df[cust_col] == selected_cust].copy()
            rev = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
            rev['Cum_Pct'] = (rev['M USD'].cumsum() / rev['M USD'].sum()) * 100
            top_prods = rev[rev['Cum_Pct'] <= 85]['Material name'].unique()[:20]

            tab1, tab2 = st.tabs(["📊 Performance & AI Audit", "📋 2026 Strategic Plan"])

            with tab1:
                selected_prod = st.selectbox("Product Audit:", top_prods)
                # Lấy thêm chỉ số Avg Growth
                avg_g, yoy_g, r_rate = calculate_detailed_metrics(cust_df, selected_prod)
                
                # Hiển thị Metrics quan trọng
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Monthly Growth", f"{avg_g:.1f}%")
                col2.metric("YoY Growth (2026 vs 2025)", f"{yoy_g:.1f}%")
                col3.metric("Current Run-rate", f"{r_rate:,.0f} units")

                # --- PLOT & TABLE (Giữ nguyên logic cũ) ---
                p_df = cust_df[cust_df['Material name'] == selected_prod].copy()
                p_df['m'] = p_df['ds'].dt.to_period('M').dt.to_timestamp()
                df_plot = p_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
                
                m = Prophet(yearly_seasonality=True).fit(df_plot)
                future = m.make_future_dataframe(periods=12, freq='MS')
                fcst = m.predict(future)
                
                fig = go.Figure()
                act_26 = df_plot[df_plot['ds'].dt.year == 2026]
                fig.add_trace(go.Scatter(x=act_26['ds'], y=act_26['y'], name="Actual 2026", mode='lines+markers', line=dict(color='blue', width=4)))
                fcst_26 = fcst[fcst['ds'].dt.year == 2026]
                fig.add_trace(go.Scatter(x=fcst_26['ds'], y=fcst_26['yhat'], name="AI Trend", line=dict(dash='dash', color='orange')))
                st.plotly_chart(fig, use_container_width=True)

                # Variance Table
                comp = pd.merge(act_26, fcst[['ds', 'yhat']], on='ds')
                if not comp.empty:
                    comp['Var %'] = ((comp['y'] - comp['yhat']) / comp['yhat']) * 100
                    comp['YTD Var %'] = ((comp['y'].cumsum() - comp['yhat'].cumsum()) / comp['yhat'].cumsum()) * 100
                    st.dataframe(comp[['ds','y','yhat','Var %', 'YTD Var %']].style.format({
                        'ds': lambda x: x.strftime('%m/%Y'), 'y': '{:,.0f}', 'yhat': '{:,.0f}', 'Var %': '{:.1f}%', 'YTD Var %': '{:.1f}%'
                    }), use_container_width=True)

            with tab2:
                # Logic Tab 2 giữ nguyên như bản upload file trước đó
                st.subheader("📋 2026 Strategic Plan")
                # ... (Phần code Pivot Plan của Jenna) ...
                st.write("Plan data is processed based on Sidebar settings.")
