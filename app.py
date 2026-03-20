import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

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
    # AI vẫn học dựa trên trend toàn bộ lịch sử 2023-2025
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
        
        growth_final = 0.0

        tab1, tab2 = st.tabs(["📊 So sánh Thực tế vs Dự báo", "📋 Kế hoạch 2026 (Sync)"])

        with tab1:
            prod_df = cust_df[cust_df['Material name'] == selected_prod].copy()
            prod_df['m'] = prod_df['ds'].dt.to_period('M').dt.to_timestamp()
            actual_all = prod_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
            
            # Lấy dữ liệu thực tế 2026 đã có
            act_2026 = actual_all[actual_all['ds'].dt.year == 2026]
            last_act_date = act_2026['ds'].max() if not act_2026.empty else pd.Timestamp('2025-12-31')
            
            res = fast_forecast(actual_all)
            if res is not None:
                # 1. Tính toán tăng trưởng dựa trên Thực tế + Dự báo
                total_25 = actual_all[actual_all['ds'].dt.year == 2025]['y'].sum()
                sum_act_26 = act_2026['y'].sum()
                fcst_future_26 = res[(res['ds'].dt.year == 2026) & (res['ds'] > last_act_date)]['yhat'].sum()
                total_26_mixed = sum_act_26 + fcst_future_26
                
                growth_final = ((total_26_mixed - total_25) / total_25 * 100) if total_25 > 0 else 0
                
                # 2. So sánh sai số dự báo (Accuracy) cho các tháng đã qua
                st.subheader("📈 Hiệu suất dự báo 2026")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Tăng trưởng dự kiến cả năm", f"{growth_final:.1f}%")
                with c2:
                    # So sánh tổng thực tế T1-T3 vs Dự báo AI cho T1-T3
                    fcst_past_26 = res[(res['ds'] >= '2026-01-01') & (res['ds'] <= last_act_date)]['yhat'].sum()
                    diff = ((sum_act_26 - fcst_past_26) / fcst_past_26 * 100) if fcst_past_26 > 0 else 0
                    st.metric("Độ lệch Thực tế vs Dự báo (YTD)", f"{diff:.1f}%", delta=f"{sum_act_26 - fcst_past_26:,.0f} pcs")

                # 3. Biểu đồ trực quan
                fig = go.Figure()
                # Đường thực tế (liên tục)
                fig.add_trace(go.Scatter(x=actual_all['ds'], y=actual_all['y'], name="Thực tế", line=dict(color='blue')))
                # Đường dự báo AI (nét đứt cho tương lai)
                fig.add_trace(go.Scatter(x=res[res['ds'] > last_act_date]['ds'], y=res[res['ds'] > last_act_date]['yhat'], 
                                         line=dict(dash='dash', color='orange'), name="Dự báo AI tiếp theo"))
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("📋 Bảng phối hợp Sản xuất - Kinh doanh")
            df_25 = cust_df[(cust_df['Material name'].isin(top_prods)) & (cust_df['ds'].dt.year == 2025)]
            base_stats = df_25.groupby(['Material name', cie_col])['Order qty.(A)'].mean().reset_index()
            
            # Thực tế 2026
            df_26_act = cust_df[(cust_df['Material name'].isin(top_prods)) & (cust_df['ds'].dt.year == 2026)]
            act_26_map = df_26_act.groupby(['Material name', cie_col, df_26_act['ds'].dt.strftime('%m/%Y')])['Order qty.(A)'].sum().to_dict()

            g_factor = 1 + (growth_final / 100)
            months_26 = [m.strftime('%m/%Y') for m in pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')]
            
            pivot_list = []
            for _, r in base_stats.iterrows():
                row = {'Sản phẩm': r['Material name'], 'CIE': r[cie_col]}
                for m_str in months_26:
                    key = (r['Material name'], r[cie_col], m_str)
                    if key in act_26_map:
                        row[m_str] = act_26_map[key] # Dùng số thực tế
                    else:
                        row[m_str] = round(r['Order qty.(A)'] * g_factor, 0) # Dùng số dự báo
                pivot_list.append(row)
            
            res_df = pd.DataFrame(pivot_list)
            st.info(f"📌 Bảng kết hợp: Dữ liệu thực tế đến {last_act_date.strftime('%m/%Y')} và dự báo cho các tháng còn lại.")
            st.dataframe(res_df.style.format("{:,.0f}", subset=months_26), use_container_width=True)
            st.download_button("📥 Xuất file kế hoạch", res_df.to_csv(index=False).encode('utf-8-sig'), "Supply_Chain_Plan_2026.csv")
