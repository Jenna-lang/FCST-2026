import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. Cấu hình & Cache dữ liệu
st.set_page_config(page_title="AI Supply Chain Advisor", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_excel('AICheck.xlsx')
    df.columns = [str(col).strip() for col in df.columns]
    df['ds'] = pd.to_datetime(df['Requested deliv. date'])
    return df

# Hàm dự báo có Cache để tránh tính toán lặp lại
@st.cache_resource
def get_forecast(data_series, periods=12):
    if len(data_series) < 2: return None
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(data_series)
    future = m.make_future_dataframe(periods=periods, freq='MS')
    forecast = m.predict(future)
    return forecast[['ds', 'yhat']]

df = load_data()

if df is not None:
    st.sidebar.header("Configuration")
    cust_list = sorted(df['Customer Name'].unique().astype(str))
    selected_cust = st.sidebar.selectbox("1. Select Customer:", cust_list)

    if selected_cust:
        cust_df = df[df['Customer Name'].astype(str) == selected_cust].copy()
        
        # Pareto calculation (Rất nhanh, không cần tối ưu)
        sales = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        sales['Cum_Pct'] = sales['M USD'].cumsum() / sales['M USD'].sum()
        top_prods = sales[sales['Cum_Pct'] <= 0.81]['Material name'].unique()
        
        selected_prod = st.selectbox("2. Select Product for Analysis:", top_prods)
        
        tab1, tab2 = st.tabs(["📊 Yearly Comparison", "📋 Full Pareto Details"])

        with tab1:
            # Chỉ tính toán cho 1 sản phẩm đang chọn (Rất nhanh)
            prod_df = cust_df[cust_df['Material name'] == selected_prod].copy()
            prod_df['m'] = prod_df['ds'].dt.to_period('M').dt.to_timestamp()
            actual_prod = prod_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
            
            res = get_forecast(actual_prod)
            if res is not None:
                # Hiển thị biểu đồ và insight (như code cũ)
                st.success(f"Analysis for {selected_prod} loaded from cache.")
                # ... (Phần code vẽ biểu đồ giữ nguyên)

        with tab2:
            st.subheader("Full Pareto Forecast Breakdown")
            # Tối ưu: Chỉ chạy khi bấm nút để tránh treo App
            if st.button("🚀 Generate/Refresh All Pareto Data"):
                all_results = []
                progress_bar = st.progress(0)
                
                for idx, p in enumerate(top_prods):
                    p_df = cust_df[cust_df['Material name'] == p].copy()
                    for c in p_df['CIE'].unique():
                        c_df = p_df[p_df['CIE'].astype(str) == c].copy()
                        c_df['m'] = c_df['ds'].dt.to_period('M').dt.to_timestamp()
                        act_c = c_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
                        
                        f_c = get_forecast(act_c)
                        if f_c is not None:
                            res26 = f_c[f_c['ds'].dt.year == 2026].copy()
                            res26['Month'] = res26['ds'].dt.strftime('%m/%Y')
                            res26['Product'] = p
                            res26['CIE'] = c
                            all_results.append(res26)
                    progress_bar.progress((idx + 1) / len(top_prods))
                
                if all_results:
                    final_table = pd.concat(all_results)
                    st.dataframe(final_table, use_container_width=True)
            else:
                st.info("Click the button above to calculate data for all Pareto products. This saves time when you only want to see the charts.")
