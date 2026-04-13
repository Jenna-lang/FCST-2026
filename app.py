import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. System Config
st.set_page_config(page_title="AI Supply Chain Advisor 2026", layout="wide")

# --- CORE LOGIC FUNCTIONS ---

def get_actual_avg_qty(df, year, quarter, prod, cie_col, cie_val):
    """Calculates monthly average for months with data (>0) in a Quarter."""
    temp = df[(df['Material name'] == prod) & 
              (df[cie_col] == cie_val) & 
              (df['ds'].dt.year == year) & 
              (df['ds'].dt.quarter == quarter)].copy()
    
    monthly_sum = temp.groupby(temp['ds'].dt.month)['Order qty.(A)'].sum()
    actual_months = monthly_sum[monthly_sum > 0]
    
    if actual_months.empty:
        return 0.0
    return actual_months.mean()

def get_quarterly_growth_logic(cust_df, prod, cie_col, cie_val):
    """Growth based on Latest 2026 Q vs 2025 same Q."""
    df_26 = cust_df[cust_df['ds'].dt.year == 2026].copy()
    if df_26.empty: return 0.0
    
    valid_26 = df_26[df_26['Order qty.(A)'] > 0]
    if valid_26.empty: return 0.0
    
    latest_q_26 = valid_26['ds'].dt.quarter.max()
    
    avg_26 = get_actual_avg_qty(cust_df, 2026, latest_q_26, prod, cie_col, cie_val)
    avg_25 = get_actual_avg_qty(cust_df, 2025, latest_q_26, prod, cie_col, cie_val)
    
    if avg_25 > 0:
        growth = (avg_26 / avg_25) - 1
        return min(max(growth, -0.5), 0.5)
    return 0.0

def process_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = [str(col).strip() for col in df.columns]
        if 'Requested deliv. date' in df.columns and 'Order qty.(A)' in df.columns:
            df['ds'] = pd.to_datetime(df['Requested deliv. date'], errors='coerce')
            df['Order qty.(A)'] = pd.to_numeric(df['Order qty.(A)
