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
        
        selected_prod = st.selectbox("2. Chọn sản phẩm:", top_prods if len(top_prods)>0 else df['Material name'].unique()[:5])
        
        tab1, tab2 = st.tabs(["📊 Trung bình cùng kỳ (AI)", "📋 Bảng ngang Pareto 2026"])

        with tab1:
            prod_df = cust_df[cust_df['Material name'] == selected_prod].copy()
            prod_df['m'] = prod_df['ds'].dt.to_period('M').dt.to_timestamp()
            actual_prod = prod_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
            
            res = fast_forecast(actual_prod)
            if res is not None:
                # Logic Trung bình cùng kỳ
                future_dates = res[res['ds'] > actual_prod['ds'].max()]
                avg_forecast = future_dates['yhat'].mean()
