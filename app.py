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
        date_col, qty_col = 'Requested deliv. date', 'Order qty.(A)'
        if date_col in df.columns and qty_col in df.columns:
            df['ds'] = pd.to_datetime(df[date_col], errors='coerce')
            df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
            df = df.dropna(subset=['ds'])
            return df
        return None
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        return None

def get_metrics(cust_df, prod_name):
    p_df = cust_df[cust_df['Material name'] == prod_name].copy()
    if p_df.empty: return 0.0, 0.0, 0.0, 0.0
    monthly = p_df.groupby(p_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
    monthly['ds_ts'] = monthly['ds'].dt.to_timestamp()
    monthly['Pct_Change'] = monthly['Order qty.(A)'].pct_change() * 100
    avg_trend = monthly['Pct_Change'].mean() if not monthly['Pct_Change'].empty else 0.0
    df_26 = monthly[monthly['ds_ts'].dt.year == 2026]
    rr_26 = df_26['Order qty.(A)'].mean() if not df_26.empty else 0.0
    # Moving Average 3 tháng gần nhất
    ma_26 = df_26['Order qty.(A)'].tail(3).mean() if not df_26.empty else 0.0
    m_26_list = df_26['ds_ts'].dt.month.tolist()
    act_25 = monthly[(monthly['ds_ts'].dt.year == 2025) & (monthly['ds_ts'].dt.month.isin(m_26_list))]['Order qty.(A)'].sum()
    yoy = (df_26['Order qty.(A)'].sum() - act_25) / act_25 if act_25 > 0 else 0.0
    return avg_trend, yoy, rr_26, ma_26

# --- GIAO DIỆN CHÍNH ---
st.sidebar.header("📁 Quản lý dữ liệu")
uploaded_file = st.sidebar.file_uploader("Tải file AICheck.xlsx", type=['xlsx'])

if uploaded_file:
    df = process_data(uploaded_file)
    if df is not None:
        # Tự động tìm cột Customer và CIE
        cust_col = [c for c in df.columns if 'Customer' in c][0] if [c for c in df.columns if 'Customer' in c] else df.columns[0]
        cie_col = [c for c in df.columns if 'CIE' in c][0] if [c for c in df.columns if 'CIE' in c] else df.columns[1]
        
        adj_growth = st.sidebar.slider("Điều chỉnh tăng trưởng (%)", -50, 50, 0)
        selected_cust = st.sidebar.selectbox("Chọn khách hàng:", ["-- Chọn --"] + sorted(df[cust_col].unique()))

        if selected_cust != "-- Chọn --":
            cust_df = df[df[cust_col] == selected_cust].copy()
            rev = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
            top_prods = rev[rev['M USD'].cumsum() / rev['M USD'].sum() <= 0.86]['Material name'].unique()[:20]

            tab1, tab2 = st.tabs(["📊 Performance Audit", "📋 2026 Strategic Plan"])

            with tab1:
                selected_prod = st.selectbox("Kiểm tra sản phẩm:", top_prods)
                trnd, yoy, rr, ma = get_metrics(cust_df, selected_prod)
                col1, col2, col3 = st.columns(3)
                col1.metric("Xu hướng lịch sử", f"{trnd:.1f}%")
                col2.metric("Tăng trưởng YoY", f"{yoy*100:.1f}%")
                col3.metric("TB trượt 3 tháng", f"{ma:,.0f}")

                # Vẽ biểu đồ AI Forecast
                p_plot = cust_df[cust_df['Material name'] == selected_prod].copy()
                m_plot = p_plot.groupby(p_plot['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
                m_plot['ds'] = m_plot['ds'].dt.to_timestamp()
                m_plot = m_plot.rename(columns={'Order qty.(A)': 'y'})
                
                model = Prophet(yearly_seasonality=True).fit(m_plot)
                fcst = model.predict(model.make_future_dataframe(periods=12, freq='MS'))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=m_plot['ds'], y=m_plot['y'], name="Thực tế", line=dict(color='blue')))
                fcst_26 = fcst[fcst['ds'].dt.year == 2026]
                fig.add_trace(go.Scatter(x=fcst_26['ds'], y=fcst_26['yhat'], name="AI Dự báo", line=dict(dash='dash', color='orange')))
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("📋 Bảng kế hoạch chi tiết 2026")
                months_26 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
                cols_26 = [m.strftime('%m/%Y') for m in months_26]
                pivot_list = []
                last_act = df[df['Order qty.(A)'] > 0]['ds'].max()

                for p in top_prods:
                    p_trnd, p_yoy, p_rr, p_ma = get_metrics(cust_df, p)
                    factor = (p_yoy if p_yoy != 0 else (p_trnd/100)) + (adj_growth / 100)
                    cies = cust_df[cust_df['Material name'] == p][cie_col].unique()
                    for c in cies:
                        row = {'Product': p, 'CIE': c}
                        for m_date in months_26:
                            m_idx, m_str = m_date.month, m_date.strftime('%m/%Y')
                            act = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2026)]['Order qty.(A)'].sum()
                            if act > 0:
                                row[m_str] = act
                            elif m_date > last_act:
                                h25 = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2025)]['Order qty.(A)'].sum()
                                base = h25 if h25 > 0 else p_ma
                                row[m_str] = round(base * (1 + factor), 0)
                            else:
                                row[m_str] = 0
                        pivot_list.append(row)
                
                if pivot_list:
                    res_df = pd.DataFrame(pivot_list)
                    st.dataframe(res_df.style.format("{:,.0f}", subset=cols_26), use_container_width=True)
