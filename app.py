import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. Cấu hình hệ thống
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
    # 2. Sidebar thiết lập cột
    st.sidebar.header("⚡ Cấu hình dữ liệu")
    cust_col = st.sidebar.selectbox("Cột khách hàng:", [c for c in df.columns if 'Customer' in c] or df.columns)
    cie_col = st.sidebar.selectbox("Cột CIE:", [c for c in df.columns if 'CIE' in c] or df.columns)
    
    cust_list = sorted(df[cust_col].unique().astype(str))
    selected_cust = st.sidebar.selectbox("1. Chọn khách hàng:", ["-- Chọn --"] + cust_list)

    if selected_cust != "-- Chọn --":
        cust_df = df[df[cust_col].astype(str) == selected_cust].copy()
        
        # Pareto 80/20 tính theo doanh số
        sales = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        sales['Cum_Pct'] = sales['M USD'].cumsum() / (sales['M USD'].sum() or 1)
        top_prods = sales[sales['Cum_Pct'] <= 0.81]['Material name'].unique()
        
        selected_prod = st.selectbox("2. Chọn sản phẩm phân tích:", top_prods if len(top_prods)>0 else df['Material name'].unique()[:5])
        
        # Khởi tạo biến tăng trưởng để dùng chung cho cả 2 Tab
        growth_final = 0.0 

        # Tạo Tab
        tab1, tab2 = st.tabs(["📊 Phân tích Tăng trưởng", "📋 Kế hoạch Sản xuất 2026"])

        # --- TAB 1: TÍNH TOÁN % TĂNG TRƯỞNG ---
        with tab1:
            prod_df = cust_df[cust_df['Material name'] == selected_prod].copy()
            prod_df['m'] = prod_df['ds'].dt.to_period('M').dt.to_timestamp()
            actual_all = prod_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
            
            # Lấy thực tế 2026 đã có
            act_2026 = actual_all[actual_all['ds'].dt.year == 2026]
            last_act_date = act_2026['ds'].max() if not act_2026.empty else pd.Timestamp('2025-12-31')
            
            res = fast_forecast(actual_all)
            if res is not None:
                # Công thức: (Thực tế đã có 2026 + Dự báo AI các tháng còn lại) vs (Tổng 2025)
                total_25 = actual_all[actual_all['ds'].dt.year == 2025]['y'].sum()
                sum_act_26 = act_2026['y'].sum()
                fcst_future_26 = res[(res['ds'].dt.year == 2026) & (res['ds'] > last_act_date)]['yhat'].sum()
                total_26_mixed = sum_act_26 + fcst_future_26
                
                growth_final = ((total_26_mixed - total_25) / total_25 * 100) if total_25 > 0 else 0
                
                st.metric("Tỉ lệ tăng trưởng dự kiến 2026", f"{growth_final:.1f}%")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=actual_all['ds'], y=actual_all['y'], name="Thực tế (Actual)"))
                fig.add_trace(go.Scatter(x=res[res['ds'] > last_act_date]['ds'], y=res[res['ds'] > last_act_date]['yhat'], 
                                         line=dict(dash='dash', color='orange'), name="AI Forecast"))
                st.plotly_chart(fig, use_container_width=True)

        # --- TAB 2: HIỂN THỊ BẢNG NGANG ĐÃ ĐỒNG BỘ ---
        with tab2:
            st.subheader("📋 Kế hoạch chi tiết theo CIE (Đã tăng trưởng)")
            df_25_raw = cust_df[(cust_df['Material name'].isin(top_prods)) & (cust_df['ds'].dt.year == 2025)]
            
            # Map dữ liệu mùa vụ 2025
            act_25_map = df_25_raw.groupby(['Material name', cie_col, df_25_raw['ds'].dt.month])['Order qty.(A)'].sum().to_dict()
            
            # Map dữ liệu thực tế 2026 đã có
            df_26_act = cust_df[(cust_df['Material name'].isin(top_prods)) & (cust_df['ds'].dt.year == 2026)]
            act_26_map = df_26_act.groupby(['Material name', cie_col, df_26_act['ds'].dt.month])['Order qty.(A)'].sum().to_dict()

            g_factor = 1 + (growth_final / 100)
            months_26 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
            columns_26 = [m.strftime('%m/%Y') for m in months_26]
            
            unique_items = df_25_raw[['Material name', cie_col]].drop_duplicates()
            pivot_list = []
            
            for _, r in unique_items.iterrows():
                row = {'Sản phẩm': r['Material name'], 'CIE': r[cie_col]}
                for m_date in months_26:
                    m_idx = m_date.month
                    m_str = m_date.strftime('%m/%Y')
                    
                    # 1. Nếu đã có thực tế 2026 (T1-T3)
                    if (r['Material name'], r[cie_col], m_idx) in act_26_map:
                        row[m_str] = act_26_map[(r['Material name'], r[cie_col], m_idx)]
                    # 2. Dự báo = (Số lượng cùng tháng năm 2025) * Hệ số tăng trưởng
                    elif (r['Material name'], r[cie_col], m_idx) in act_25_map:
                        row[m_str] = round(act_25_map[(r['Material name'], r[cie_col], m_idx)] * g_factor, 0)
                    else:
                        row[m_str] = 0
                pivot_list.append(row)
            
            res_df = pd.DataFrame(pivot_list)
            st.info(f"📌 Ghi chú: Các tháng T4-T12 đã được nhân hệ số {g_factor:.2f} dựa trên sản lượng cùng kỳ 2025.")
            st.dataframe(res_df.style.format("{:,.0f}", subset=columns_26), use_container_width=True)
            st.download_button("📥 Tải kế hoạch CSV", res_df.to_csv(index=False).encode('utf-8-sig'), "Plan_2026_Final.csv")
