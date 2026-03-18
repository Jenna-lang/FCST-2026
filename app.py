import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="AI Supply Chain Advisor", layout="wide")
st.title("🤖 AI Strategic Advisor: History & 2026 Forecast")

@st.cache_data
def load_data():
    df = pd.read_excel('AICheck.xlsx')
    df.columns = [str(col).strip() for col in df.columns]
    df['ds'] = pd.to_datetime(df['Requested deliv. date'])
    return df

try:
    df = load_data()
    
    # 2. Sidebar Settings
    st.sidebar.header("Data Configuration")
    cust_col = st.sidebar.selectbox("Customer Name Column:", df.columns)
    cie_col = st.sidebar.selectbox("CIE Color Column:", df.columns)
    
    # Per-Customer Pareto Filter
    cust_list = sorted(df[cust_col].unique().astype(str))
    selected_cust = st.selectbox("1. Select Customer:", cust_list)

    if selected_cust:
        cust_df = df[df[cust_col].astype(str) == selected_cust].copy()
        
        # Pareto Logic (80/20) for this specific customer
        sales = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        sales['Cum_Pct'] = sales['M USD'].cumsum() / sales['M USD'].sum()
        top_prods = sales[sales['Cum_Pct'] <= 0.81]['Material name'].unique()
        
        selected_prod = st.selectbox(f"2. Select Top Product for {selected_cust}:", top_prods)

        if selected_prod:
            prod_df = cust_df[cust_df['Material name'] == selected_prod].copy()
            cie_options = sorted(prod_df[cie_col].unique().astype(str))
            selected_cies = st.multiselect("3. Select CIE Color Codes:", cie_options, default=cie_options[:1])

            if selected_cies:
                fig = go.Figure()
                forecast_results = []
                ai_insights = []
                
                for cie in selected_cies:
                    cie_df = prod_df[prod_df[cie_col].astype(str) == cie].copy()
                    cie_df['m'] = cie_df['ds'].dt.to_period('M').dt.to_timestamp()
                    actual = cie_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
                    
                    if len(actual
