import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. Cấu hình & Cache dữ liệu (Tối ưu RAM)
st.set_page_config(page_title="AI Supply Chain Advisor", layout="wide")

@st.cache_data(ttl=3600) # Lưu dữ liệu trong 1 giờ
def load_data():
    try:
        df = pd.read_excel('AICheck.xlsx')
        df.columns = [str(col).strip() for col in df.columns]
        df['ds'] = pd.to_datetime(df['Requested deliv. date'])
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Hàm dự báo siêu nhanh (Tắt các tính năng thừa để giảm tải CPU)
@st.cache_resource
def fast_forecast(data_series):
    if len(data_series) < 2: return None
    # Tắt tính năng dự báo theo ngày/tuần vì dữ liệu LED thường theo tháng
    m = Prophet(
        yearly_seasonality=True, 
        weekly_seasonality=False, 
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    m.fit(data_series)
    future = m.make_future_dataframe(periods=12, freq='MS')
    forecast = m.predict(future)
    return forecast[['ds', 'yhat']]

df = load_data()

if df is not None:
    # 2. Sidebar rút gọn
    st.sidebar.header("⚡ Fast Config")
    cust_col = st.sidebar.selectbox("Customer Column:", df.columns)
    cie_col = st.sidebar.selectbox("CIE Column:", df.columns)
    
    cust_list = sorted(df[cust_col].unique().astype(str))
    selected_cust = st.sidebar.selectbox("1. Select Customer:", cust_list)

    if selected_cust:
        cust_df = df[df[cust_col].astype(str) == selected_cust].copy()
        
        # Pareto calculation (Hàm vector của Pandas cực nhanh)
        sales = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        sales['Cum_Pct'] = sales['M USD'].cumsum() / sales['M USD'].sum()
        top_prods = sales[sales['Cum_Pct'] <= 0.81]['Material name'].unique()
        
        selected_prod = st.selectbox("2. Select Product for Quick Analysis:", top_prods)
        
        tab1, tab2 = st.tabs(["📊 Yearly Comparison", "📋 Full Pareto Details (Slow)"])

        with tab1:
            # CHỈ TÍNH TOÁN CHO SẢN PHẨM ĐANG CHỌN
            prod_df = cust_df[cust_df['Material name'] == selected_prod].copy()
            prod_df['m'] = prod_df['ds'].dt.to_period('M').dt.to_timestamp()
            actual_prod = prod_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
            
            res = fast_forecast(actual_prod)
            
            if res is not None:
                # Tính toán tăng trưởng trung bình năm
                avg_25 = actual_prod[actual_prod['ds'].dt.year == 2025]['y'].sum()
                avg_26 = res[res['ds'].dt.year == 2026]['yhat'].sum()
                growth = ((avg_26 - avg_25) / avg_25 * 100) if avg_25 > 0 else 0
                
                color = "green" if growth > 0 else "red"
                st.markdown(f"### Yearly Growth: :{color}[{growth:.1f}%]")
                
                # Vẽ biểu đồ 2 cột (Chart & Bar)
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig_l = go.Figure()
                    fig_l.add_trace(go.Scatter(x=actual_prod['ds'], y=actual_prod['y'], name="Actual"))
                    f_only = res[res['ds'] > actual_prod['ds'].max()]
                    fig_l.add_trace(go.Scatter(x=f_only['ds'], y=f_only['yhat'], line=dict(dash='dash'), name="2026 FCST"))
                    st.plotly_chart(fig_l, use_container_width=True)
                with c2:
                    fig_b = go.Figure(go.Bar(x=['2025', '2026'], y=[avg_25, avg_26], marker_color=[None, color]))
                    fig_b.update_layout(height=400, title="Total Volume Comparison")
                    st.plotly_chart(fig_b, use_container_width=True)

        with tab2:
            st.warning("⚠️ Calculating all Pareto items might take a minute.")
            # CHỈ CHẠY KHI NHẤN NÚT - Đây là chìa khóa để App không bị lag
            if st.button("🚀 Run Full Pareto Analysis"):
                all_results = []
                bar = st.progress(0)
                for i, p in enumerate(top_prods):
                    p_df = cust_df[cust_df['Material name'] == p].copy()
                    for c in p_df[cie_col].unique():
                        c_df = p_df[p_df[cie_col].astype(str) == c].copy()
                        c_df['m'] = c_df['ds'].dt.to_period('M').dt.to_timestamp()
                        act_c = c_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
                        
                        f_c = fast_forecast(act_c)
                        if f_c is not None:
                            r26 = f_c[f_c['ds'].dt.year == 2026].copy()
                            r26['Month'] = r26['ds'].dt.strftime('%m/%Y')
                            r26['Product'] = p
                            r26['CIE'] = c
                            all_results.append(r26)
                    bar.progress((i + 1) / len(top_prods))
                
                if all_results:
                    final = pd.concat(all_results).rename(columns={'yhat': 'Qty (Pcs)'})
                    st.dataframe(final.style.format("{:,.0f}", subset=['Qty (Pcs)']), use_container_width=True)
                    st.download_button("📥 Download Report", final.to_csv(index=False).encode('utf-8-sig'), "Pareto_2026.csv")
