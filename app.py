import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. System Setup
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
        st.error(f"File Read Error: {e}")
        return None

def get_precise_metrics(cust_df, prod_name, cie_name=None):
    if cie_name:
        p_df = cust_df[(cust_df['Material name'] == prod_name) & (cust_df.iloc[:, 1] == cie_name)].copy()
    else:
        p_df = cust_df[cust_df['Material name'] == prod_name].copy()
        
    if p_df.empty: return 0.0, 0.0, 0.0
    
    monthly = p_df.groupby(p_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
    monthly['ds_ts'] = monthly['ds'].dt.to_timestamp()
    
    monthly['Pct_Change'] = monthly['Order qty.(A)'].pct_change() * 100
    avg_trend = monthly['Pct_Change'].mean() if not monthly['Pct_Change'].empty else 0.0
    
    df_26 = monthly[monthly['ds_ts'].dt.year == 2026]
    ma_26 = df_26['Order qty.(A)'].tail(3).mean() if not df_26.empty else 0.0
    
    m_26_list = df_26['ds_ts'].dt.month.tolist()
    act_25 = monthly[(monthly['ds_ts'].dt.year == 2025) & (monthly['ds_ts'].dt.month.isin(m_26_list))]['Order qty.(A)'].sum()
    yoy_raw = (df_26['Order qty.(A)'].sum() - act_25) / act_25 if act_25 > 0 else 0.0
    yoy_final = min(yoy_raw, 0.5) # Growth Cap to prevent abnormal spikes
    
    return avg_trend, yoy_final, ma_26

# --- MAIN INTERFACE (ALL ENGLISH) ---
st.sidebar.header("📁 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload AICheck.xlsx", type=['xlsx'])

if uploaded_file:
    df = process_data(uploaded_file)
    if df is not None:
        c_cols = [c for c in df.columns if 'Customer' in c]
        cie_col_name = df.columns[1] 
        cust_col = c_cols[0] if c_cols else df.columns[0]
        
        adj_growth = st.sidebar.slider("Manual Adjustment (%)", -50, 50, 0)
        selected_cust = st.sidebar.selectbox("Select Customer:", ["-- Select --"] + sorted(df[cust_col].unique()))

        if selected_cust != "-- Select --":
            cust_df = df[df[cust_col] == selected_cust].copy()
            
            # --- PARETO 85% LOGIC ---
            rev = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
            rev['CumSum'] = rev['M USD'].cumsum() / rev['M USD'].sum()
            top_prods = rev[rev['CumSum'] <= 0.86]['Material name'].unique()[:20]

            tab1, tab2 = st.tabs(["📊 Performance Audit", "📋 2026 Strategic Plan"])

            with tab1:
                selected_prod = st.selectbox("Product Audit:", top_prods)
                trnd, yoy, ma = get_precise_metrics(cust_df, selected_prod)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Historical Trend", f"{trnd:.1f}%")
                m2.metric("YoY Growth", f"{
