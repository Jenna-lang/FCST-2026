import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. System Config
st.set_page_config(page_title="AI Supply Chain Advisor 2026", layout="wide")

# --- CORE LOGIC FUNCTIONS ---
def get_actual_avg_qty(df, year, quarter, prod, cie_col, cie_val):
    temp = df[(df['Material name'] == prod) & 
              (df[cie_col] == cie_val) & 
              (df['ds'].dt.year == year) & 
              (df['ds'].dt.quarter == quarter)].copy()
    monthly_sum = temp.groupby(temp['ds'].dt.month)['Order qty.(A)'].sum()
    actual_months = monthly_sum[monthly_sum > 0]
    return actual_months.mean() if not actual_months.empty else 0.0

def process_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = [str(col).strip() for col in df.columns]
        if 'Requested deliv. date' in df.columns:
            df['ds'] = pd.to_datetime(df['Requested deliv. date'], errors='coerce')
            df['Order qty.(A)'] = pd.to_numeric(df['Order qty.(A)'], errors='coerce').fillna(0)
            return df.dropna(subset=['ds'])
        return None
    except: return None

# --- MAIN UI ---
st.sidebar.header("📁 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload AICheck.xlsx", type=['xlsx'])

if uploaded_file:
    df = process_data(uploaded_file)
    if df is not None:
        all_cols = df.columns.tolist()
        cust_col = st.sidebar.selectbox("Customer Column:", all_cols, index=0)
        cie_col = st.sidebar.selectbox("CIE / Color Code Column:", all_cols, index=1)
        adj_growth = st.sidebar.slider("Manual Growth Adjustment (%)", -50, 50, 0)
        selected_cust = st.sidebar.selectbox("Select Target Customer:", ["-- Select --"] + sorted(df[cust_col].unique().tolist()))

        if selected_cust != "-- Select --":
            cust_df = df[df[cust_col] == selected_cust].copy()
            rev = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
            rev['Cum%'] = rev['M USD'].cumsum() / rev['M USD'].sum()
            top_prods = rev[rev['Cum%'] <= 0.86]['Material name'].unique()

            # --- PRE-CALCULATE ALL VARIANCES (Fixing the data loss issue) ---
            auto_adjustments = {}
            with st.spinner('🔄 AI is calculating performance metrics...'):
                for p in top_prods:
                    p_data = cust_df[cust_df['Material name'] == p].groupby(cust_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
                    p_data['ds'] = p_data['ds'].dt.to_timestamp()
                    p_data = p_data.rename(columns={'Order qty.(A)': 'y'})
                    if len(p_data) > 2:
                        m = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False).fit(p_data)
                        f = m.predict(m.make_future_dataframe(periods=1, freq='MS'))
                        v_26 = pd.merge(p_data[p_data['ds'].dt.year == 2026], f[['ds', 'yhat']], on='ds')
                        auto_adjustments[p] = ((v_26['y'] - v_26['yhat']) / v_26['yhat']).mean() if not v_26.empty else 0.0

            # --- TẠO TAB ---
            tab1, tab2, tab3 = st.tabs(["📊 Performance Audit", "📋 2026 Strategic Plan", "🧪 Model Testing"])

            with tab1:
                st.subheader("🎯 Variance Audit & Visuals")
                selected_prod = st.selectbox("Select Product to Audit:", top_prods)
                
                # Biểu đồ chi tiết cho mã đã chọn
                p_plot = cust_df[cust_df['Material name'] == selected_prod].groupby(cust_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
                p_plot['ds'] = p_plot['ds'].dt.to_timestamp()
                p_plot = p_plot.rename(columns={'Order qty.(A)': 'y'})
                
                m_audit = Prophet(yearly_seasonality=True).fit(p_plot)
                fcst_audit = m_audit.predict(m_audit.make_future_dataframe(periods=12, freq='MS'))
                fcst_26_audit = fcst_audit[fcst_audit['ds'].dt.year == 2026].copy()
                v_df_audit = pd.merge(p_plot[p_plot['ds'].dt.year == 2026], fcst_26_audit[['ds', 'yhat']], on='ds')
                
                cur_v = auto_adjustments.get(selected_prod, 0.0)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=p_plot['ds'], y=p_plot['y'], name="Actual (Thực tế)"))
                fig.add_trace(go.Scatter(x=fcst_26_audit['ds'], y=fcst_26_audit['yhat']*(1+cur_v), name="Adjusted FCST", line=dict(dash='dash', color='orange')))
                st.plotly_chart(fig, use_container_width=True)

                # Bảng Variance có dòng AVERAGE
                if not v_df_audit.empty:
                    v_table = v_df_audit
