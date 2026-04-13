import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. System Configuration
st.set_page_config(page_title="AI Supply Chain Advisor 2026", layout="wide")

if 'adj_dict' not in st.session_state:
    st.session_state.adj_dict = {}

# --- HELPER FUNCTIONS ---
def find_col(df, keywords):
    for k in keywords:
        for col in df.columns:
            if k.lower() in str(col).lower():
                return col
    return df.columns[0]

def get_actual_avg_qty(df, year, quarter, prod, cie_col, cie_val, mat_col):
    temp = df[(df[mat_col] == prod) & 
              (df[cie_col].astype(str) == str(cie_val)) & 
              (df['ds'].dt.year == year) & 
              (df['ds'].dt.quarter == quarter)].copy()
    monthly_sum = temp.groupby(temp['ds'].dt.month)['Order qty.(A)'].sum()
    actual_months = monthly_sum[monthly_sum > 0]
    return actual_months.mean() if not actual_months.empty else 0.0

def get_quarterly_growth_logic(cust_df, prod, cie_col, cie_val, mat_col):
    df_26 = cust_df[cust_df['ds'].dt.year == 2026].copy()
    valid_26 = df_26[df_26['Order qty.(A)'] > 0]
    if valid_26.empty: return 0.0
    latest_q_26 = valid_26['ds'].dt.quarter.max()
    avg_26 = get_actual_avg_qty(cust_df, 2026, latest_q_26, prod, cie_col, cie_val, mat_col)
    avg_25 = get_actual_avg_qty(cust_df, 2025, latest_q_26, prod, cie_col, cie_val, mat_col)
    return min(max((avg_26 / avg_25) - 1, -0.5), 0.5) if avg_25 > 0 else 0.0

# --- DATA PROCESSING ---
def process_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = [str(col).strip() for col in df.columns]
        date_c = find_col(df, ['date', 'requested'])
        qty_c = find_col(df, ['qty', 'order'])
        df['ds'] = pd.to_datetime(df[date_c], errors='coerce')
        df['Order qty.(A)'] = pd.to_numeric(df[qty_c], errors='coerce').fillna(0)
        return df.dropna(subset=['ds'])
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# --- UI MAIN ---
uploaded_file = st.sidebar.file_uploader("Upload AICheck.xlsx", type=['xlsx'])

if uploaded_file:
    df = process_data(uploaded_file)
    if df is not None:
        all_cols = df.columns.tolist()
        cust_col = st.sidebar.selectbox("Customer Column:", all_cols, index=all_cols.index(find_col(df, ['customer'])))
        cie_col = st.sidebar.selectbox("CIE / Item Code Column:", all_cols, index=all_cols.index(find_col(df, ['cie', 'item', 'code'])))
        mat_col = find_col(df, ['material', 'product'])
        
        selected_cust = st.sidebar.selectbox("Select Target Customer:", ["-- Select --"] + sorted(df[cust_col].unique().tolist()))

        if selected_cust != "-- Select --":
            cust_df = df[df[cust_col] == selected_cust].copy()
            rev_c = find_col(df, ['m usd', 'revenue', 'value'])
            rev = cust_df.groupby(mat_col)[rev_c].sum().sort_values(ascending=False).reset_index()
            top_prods = rev[rev[rev_c].cumsum() / rev[rev_c].sum() <= 0.86][mat_col].unique()[:50]

            tab1, tab2 = st.tabs(["📊 Performance Audit", "📋 2026 Strategic Plan"])

            with tab1:
                st.subheader("🔍 Individual Product Tuning")
                selected_prod = st.selectbox("Select Product:", top_prods)
                
                curr_adj = st.session_state.adj_dict.get(selected_prod, 0)
                new_adj = st.number_input(f"Manual Adjustment for {selected_prod} (%)", value=curr_adj)
                st.session_state.adj_dict[selected_prod] = new_adj
                
                # Variance Logic
                p_data = cust_df[cust_df[mat_col] ==
