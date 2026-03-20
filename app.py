import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. Cấu hình hệ thống
st.set_page_config(page_title="AI Supply Chain Advisor 2026", layout="wide")

# --- HÀM XỬ LÝ DỮ LIỆU ---
def process_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = [str(col).strip() for col in df.columns]
        date_col = 'Requested deliv. date'
        if date_col in df.columns:
            # Đảm bảo định dạng datetime chuẩn
            df['ds'] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=['ds'])
            return df
        else:
            st.error(f"Thiếu cột '{date_col}' trong file!")
            return None
    except Exception as e:
        st.error(f"Lỗi định dạng file: {e}")
        return None

def calculate_metrics(cust_df, prod_name):
    # Lọc đúng sản phẩm
    p_df = cust_df[cust_df['Material name'] == prod_name].copy()
    
    # Tạo cột tháng/năm để nhóm chính xác
    p_df['year'] = p_df['ds'].dt.year
    p_df['month'] = p_df['ds'].dt.month
    
    # 1. Tính Average Monthly Growth (Toàn bộ lịch sử)
    monthly_all = p_df.groupby(p_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
    monthly_all['Pct_Change'] = monthly_all['Order qty.(A)'].pct_change() * 100
    avg_growth = monthly_all['Pct_Change'].mean()
    
    # 2. Tính số thực tế 2026 (Chỉ lấy năm 2026)
    df_2026 = p_df[p_df['year'] == 2026]
    act_2026_series = df_2026.groupby('month')['Order qty.(A)'].sum()
    
    # 3. Tính YoY Growth (So sánh tháng có thực tế của 2026 với cùng tháng đó của 2025)
    active_months_26 = act_2026_series.index.tolist()
    if not active_months_26: 
        return avg_growth, 0.0, 0.0
    
    df_2025_same = p_df[(p_df['year'] == 2025) & (p_df['month'].isin(active_months_26))]
    act_2025_sum = df_2025_same['Order qty.(A)'].sum()
    act_2026_sum = act_2026_series.sum()
    
    yoy_growth = (act_2026_sum - act_2025_sum) / act_2025_sum if act_2025_sum > 0 else 0.0
    run_rate_26 = act_2026_series.mean()
    
    return avg_growth, yoy_growth, run_rate_26

# --- SIDEBAR ---
st.sidebar.header("📁 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload AICheck.xlsx file", type=['xlsx'])

if uploaded_file is not None:
    df = process_data(uploaded_file)
    if df is not None:
        cust_col = st.sidebar.selectbox("Customer:", [c for c in df.columns if 'Customer' in c] or [df.columns[0]])
        cie_col = st.sidebar.selectbox("CIE/Color:", [c for c in df.columns if 'CIE' in c] or [df.columns[1]])
        adj_growth = st.sidebar.slider("Manual Adjustment (%)", -50, 50, 0)
        selected_cust = st.sidebar.selectbox("Select Customer:", ["-- Select --"] + sorted(df[cust_col].unique()))

        if selected_cust != "-- Select --":
            cust_df = df[df[cust_col] == selected_cust].copy()
            rev = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
            top_prods = rev[rev['M USD'].cumsum() / rev['M USD'].sum() <= 0.86]['Material name'].unique()[:20]

            tab1, tab2 = st.tabs(["📊 Performance & AI Audit", "📋 2026 Strategic Plan"])

            with tab1:
                selected_prod = st.selectbox("Product Audit:", top_prods)
                avg_g, yoy_g, r_rate = calculate_metrics(cust_df, selected_prod)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Avg Monthly Growth", f"{avg_g:.1f}%")
                m2.metric("YoY Growth (Actual 26 vs 25)", f"{yoy_g*100:.1f}%")
                m3.metric("Current Run-rate (2026)", f"{r_rate:,.0f}")

                # Logic Plot - Đảm bảo lấy đúng năm 2026 cho đường Actual
                p_df = cust_df[cust_df['Material name'] == selected_prod].copy()
                df_monthly = p_df.groupby(p_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
                df_monthly['ds'] = df_monthly['ds'].dt.to_timestamp()
                df_monthly = df_monthly.rename(columns={'Order qty.(A)': 'y'})

                # Prophet
                m = Prophet(yearly_seasonality=True).fit(df_monthly)
                future = m.make_future_dataframe(periods=12, freq='MS')
                fcst = m.predict(future)

                fig = go.Figure()
                # Actual 2026
                act_26 = df_monthly[df_monthly['ds'].dt.year == 2026]
                fig.add_trace(go.Scatter(x=act_26['ds'], y=act_26['y'], name="Actual 2026", mode='lines+markers', line=dict(color='blue', width=4)))
                # AI Trend 2026
                fcst_26 = fcst[fcst['ds'].dt.year == 2026]
                fig.add_trace(go.Scatter(x=fcst_26['ds'], y=fcst_26['yhat'], name="AI Trend", line=dict(dash='dash', color='orange')))
                st.plotly_chart(fig, use_container_width=True)

                # Bảng Variance
                comp = pd.merge(act_26, fcst[['ds', 'yhat']], on='ds')
                if not comp.empty:
                    comp['Var %'] = ((comp['y'] - comp['yhat']) / comp['yhat']) * 100
                    st.subheader("🔢 Chi tiết thực tế 2026")
                    st.dataframe(comp[['ds','y','yhat','Var %']].style.format({'ds': lambda x: x.strftime('%m/%Y'), 'y': '{:,.0f}', 'yhat': '{:,.0f}', 'Var %': '{:.1f}%'}))

            with tab2:
                # Logic Strategic Plan (Dùng hàm calculate_metrics đã sửa)
                st.subheader("📋 2026 Strategic Plan")
                # ... (Giữ nguyên logic Pivot đã tối ưu của Jenna) ...
