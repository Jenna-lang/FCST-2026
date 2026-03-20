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
            df['ds'] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=['ds'])
            return df
        else:
            st.error(f"Không tìm thấy cột '{date_col}'!")
            return None
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        return None

def calculate_metrics_all_years(cust_df, prod_name):
    # Lọc sản phẩm
    p_df = cust_df[cust_df['Material name'] == prod_name].copy()
    
    # Gom nhóm theo tháng cho TOÀN BỘ lịch sử (2023, 2024, 2025, 2026...)
    monthly_all = p_df.groupby(p_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
    monthly_all['ds_ts'] = monthly_all['ds'].dt.to_timestamp()
    
    # 1. Avg Growth (Tính trên toàn bộ lịch sử từ 2023)
    monthly_all['Pct_Change'] = monthly_all['Order qty.(A)'].pct_change() * 100
    avg_growth = monthly_all['Pct_Change'].mean()
    
    # 2. Số thực tế 2026
    act_2026_df = monthly_all[monthly_all['ds_ts'].dt.year == 2026]
    act_2026_sum = act_2026_df['Order qty.(A)'].sum()
    
    # 3. YoY Growth (So sánh phần đã có của 2026 với cùng kỳ 2025)
    months_active_26 = act_2026_df['ds_ts'].dt.month.tolist()
    act_2025_same = monthly_all[(monthly_all['ds_ts'].dt.year == 2025) & 
                                (monthly_all['ds_ts'].dt.month.isin(months_active_26))]['Order qty.(A)'].sum()
    
    yoy_growth = (act_2026_sum - act_2025_same) / act_2025_same if act_2025_same > 0 else 0.0
    run_rate = act_2026_df['Order qty.(A)'].mean() if not act_2026_df.empty else 0.0
    
    return avg_growth, yoy_growth, run_rate

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
            
            # Pareto 85% Doanh thu
            rev = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
            top_prods = rev[rev['M USD'].cumsum() / rev['M USD'].sum() <= 0.86]['Material name'].unique()[:20]

            tab1, tab2 = st.tabs(["📊 Performance & AI Audit", "📋 2026 Strategic Plan"])

            with tab1:
                selected_prod = st.selectbox("Product Audit:", top_prods)
                avg_g, yoy_g, r_rate = calculate_metrics_all_years(cust_df, selected_prod)
                
                # Hiển thị Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Avg Monthly Growth (History)", f"{avg_g:.1f}%")
                m2.metric("YoY Growth (26 vs 25)", f"{yoy_g*100:.1f}%")
                m3.metric("Current Run-rate 2026", f"{r_rate:,.0f}")

                # BIỂU ĐỒ TOÀN BỘ LỊCH SỬ (2023 -> 2026)
                p_df = cust_df[cust_df['Material name'] == selected_prod].copy()
                df_monthly = p_df.groupby(p_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
                df_monthly['ds'] = df_monthly['ds'].dt.to_timestamp()
                df_monthly = df_monthly.rename(columns={'Order qty.(A)': 'y'})

                # AI Prophet Training với toàn bộ data
                m = Prophet(yearly_seasonality=True).fit(df_monthly)
                future = m.make_future_dataframe(periods=12, freq='MS')
                fcst = m.predict(future)

                fig = go.Figure()
                # 1. Lịch sử (2023, 2024, 2025)
                history = df_monthly[df_monthly['ds'].dt.year < 2026]
                fig.add_trace(go.Scatter(x=history['ds'], y=history['y'], name="History (2023-2025)", line=dict(color='lightgray')))
                
                #
