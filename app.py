import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. Cấu hình & Tải dữ liệu
st.set_page_config(page_title="AI Supply Chain Advisor", layout="wide")

@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_excel('AICheck.xlsx')
        df.columns = [str(col).strip() for col in df.columns]
        df['ds'] = pd.to_datetime(df['Requested deliv. date'], errors='coerce')
        df = df.dropna(subset=['ds'])
        return df
    except Exception as e:
        st.error(f"Lỗi tải file: {e}")
        return None

@st.cache_resource
def fast_forecast(data_series):
    if len(data_series) < 2: return None
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(data_series)
    future = m.make_future_dataframe(periods=12, freq='MS')
    fcst = m.predict(future)
    return fcst[['ds', 'yhat']]

df = load_data()

if df is not None:
    # 2. Sidebar Thiết lập
    st.sidebar.header("⚡ Cấu hình")
    cust_col = st.sidebar.selectbox("Cột khách hàng:", [c for c in df.columns if 'Customer' in c] or df.columns)
    cie_col = st.sidebar.selectbox("Cột CIE:", [c for c in df.columns if 'CIE' in c] or df.columns)
    
    cust_list = sorted(df[cust_col].unique().astype(str))
    selected_cust = st.sidebar.selectbox("1. Chọn khách hàng:", ["-- Chọn --"] + cust_list)

    if selected_cust != "-- Chọn --":
        cust_df = df[df[cust_col].astype(str) == selected_cust].copy()
        
        # Pareto 80/20
        sales = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        sales['Cum_Pct'] = sales['M USD'].cumsum() / (sales['M USD'].sum() or 1)
        top_prods = sales[sales['Cum_Pct'] <= 0.81]['Material name'].unique()
        
        selected_prod = st.selectbox("2. Chọn sản phẩm phân tích:", top_prods if len(top_prods)>0 else df['Material name'].unique()[:5])
        
        # Biến quan trọng để đồng bộ 2 Tab
        growth_final = 0.0 

        tab1, tab2 = st.tabs(["📊 Phân tích Thực tế vs Dự báo", "📋 Kế hoạch 2026 (Đồng bộ mùa vụ)"])

        # --- TAB 1: LOGIC DỰ BÁO ---
        with tab1:
            prod_df = cust_df[cust_df['Material name'] == selected_prod].copy()
            prod_df['m'] = prod_df['ds'].dt.to_period('M').dt.to_timestamp()
            actual_all = prod_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
            
            # Tách dữ liệu thực tế 2026 (T1-T3)
            act_2026 = actual_all[actual_all['ds'].dt.year == 2026]
            last_act_date = act_2026['ds'].max() if not act_2026.empty else pd.Timestamp('2025-12-31')
            
            res = fast_forecast(actual_all)
            if res is not None:
                # Tính tăng trưởng: (Thực tế 2026 + Dự báo còn lại 2026) vs (Tổng 2025)
                total_25 = actual_all[actual_all['ds'].dt.year == 2025]['y'].sum()
                sum_act_26 = act_2026['y'].sum()
                fcst_future_26 = res[(res['ds'].dt.year == 2026) & (res['ds'] > last_act_date)]['yhat'].sum()
                total_26_mixed = sum_act_26 + fcst_future_26
                
                growth_final = ((total_26_mixed - total_25) / total_25 * 100) if total_25 > 0 else 0
                
                st.metric("Tăng trưởng dự kiến 2026", f"{growth_final:.1f}%")
                
                # Biểu đồ xu hướng
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=actual_all['ds'], y=actual_all['y'], name="Thực tế (Actual)"))
                fig.add_trace
