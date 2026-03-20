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
            st.error(f"Thiếu cột '{date_col}' trong file!")
            return None
    except Exception as e:
        st.error(f"Lỗi định dạng: {e}")
        return None

def calculate_advanced_metrics(cust_df, prod_name):
    p_df = cust_df[cust_df['Material name'] == prod_name].copy()
    if p_df.empty: return 0, 0, 0
    
    # Gom nhóm theo tháng cho toàn bộ lịch sử (từ 2023)
    monthly_all = p_df.groupby(p_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
    monthly_all['ds_ts'] = monthly_all['ds'].dt.to_timestamp()
    
    # 1. Avg Monthly Growth
    monthly_all['Pct_Change'] = monthly_all['Order qty.(A)'].pct_change() * 100
    avg_growth = monthly_all['Pct_Change'].mean()
    
    # 2. Metrics 2026
    df_2026 = monthly_all[monthly_all['ds_ts'].dt.year == 2026]
    act_2026_sum = df_2026['Order qty.(A)'].sum()
    run_rate_26 = df_2026['Order qty.(A)'].mean() if not df_2026.empty else 0
    
    # 3. YoY Growth (2026 vs 2025 cùng kỳ)
    months_active_26 = df_2026['ds_ts'].dt.month.tolist()
    act_2025_same = monthly_all[(monthly_all['ds_ts'].dt.year == 2025) & 
                                (monthly_all['ds_ts'].dt.month.isin(months_active_26))]['Order qty.(A)'].sum()
    yoy_growth = (act_2026_sum - act_2025_same) / act_2025_same if act_2025_same > 0 else 0.0
    
    return avg_growth, yoy_growth, run_rate_26

# --- SIDEBAR ---
st.sidebar.header("📁 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload AICheck.xlsx", type=['xlsx'])

if uploaded_file:
    df = process_data(uploaded_file)
    if df is not None:
        # Tự động nhận diện cột
        cust_col = st.sidebar.selectbox("Customer:", [c for c in df.columns if 'Customer' in c] or [df.columns[0]])
        cie_col = st.sidebar.selectbox("CIE/Color:", [c for c in df.columns if 'CIE' in c] or [df.columns[1]])
        adj_growth = st.sidebar.slider("Manual Adjustment (%)", -50, 50, 0)
        selected_cust = st.sidebar.selectbox("Select Customer:", ["-- Select --"] + sorted(df[cust_col].unique()))

        if selected_cust != "-- Select --":
            cust_df = df[df[cust_col] == selected_cust].copy()
            
            # Pareto 85% Logic
            rev = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
            rev['Cum_Pct'] = (rev['M USD'].cumsum() / rev['M USD'].sum()) * 100
            top_prods = rev[rev['Cum_Pct'] <= 86]['Material name'].unique()[:20]

            tab1, tab2 = st.tabs(["📊 Performance Audit", "📋 2026 Strategic Plan"])

            with tab1:
                selected_prod = st.selectbox("Product Audit:", top_prods)
                avg_g, yoy_g, r_rate = calculate_advanced_metrics(cust_df, selected_prod)
                
                # Hiển thị Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg Monthly Growth", f"{avg_g:.1f}%")
                c2.metric("YoY Growth (26 vs 25)", f"{yoy_g*100:.1f}%")
                c3.metric("Current Run-rate 2026", f"{r_rate:,.0f}")

                # BIỂU ĐỒ LỊCH SỬ (2023 - 2026)
                p_df_prod = cust_df[cust_df['Material name'] == selected_prod].copy()
                df_plot = p_df_prod.groupby(p_df_prod['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
                df_plot['ds'] = df_plot['ds'].dt.to_timestamp()
                df_plot = df_plot.rename(columns={'Order qty.(A)': 'y'})

                # Huấn luyện Prophet
                m = Prophet(yearly_seasonality=True).fit(df_plot)
                future = m.make_future_dataframe(periods=12, freq='MS')
                fcst = m.predict(future)
