import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

# Basic Page Configuration
st.set_page_config(page_title="2026 Forecast & CIE Analysis", layout="wide")
st.title("🚀 Supply Chain Forecast & CIE Analysis 2026")

@st.cache_data
def load_data():
    # Loading data and cleaning column names
    df = pd.read_excel('AICheck.xlsx')
    df.columns = [str(col).strip() for col in df.columns]
    # Ensure delivery date is in datetime format
    df['ds'] = pd.to_datetime(df['Requested deliv. date'])
    return df

try:
    df = load_data()
    
    # --- SIDEBAR SETTINGS ---
    st.sidebar.header("⚙️ Configuration")
    # Allows you to pick the correct column for Customer Names and CIE Color Codes
    cust_col = st.sidebar.selectbox("Customer Name Column:", df.columns)
    cie_col = st.sidebar.selectbox("CIE Color Code Column:", df.columns)
    
    # Pareto Filtering (Top 85% of Revenue)
    summary = df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
    summary['Cum_Pct'] = summary['M USD'].cumsum() / summary['M USD'].sum()
    pareto_list = summary[summary['Cum_Pct'] <= 0.85]['Material name'].unique()

    selected_prod = st.sidebar.selectbox("1. Select Product:", pareto_list)

    if selected_prod:
        prod_df = df[df['Material name'] == selected_prod].copy()
        
        # --- SELECTION FILTERS ---
        st.subheader(f"🔍 Analyzing: {selected_prod}")
        
        # Customer Filter
        cust_list = sorted([str(x) for x in prod_df[cust_col].unique()])
        selected_cust = st.selectbox("2. Select Customer:", ["ALL Customers"] + cust_list)
        
        base_df = prod_df if selected_cust == "ALL Customers" else prod_df[prod_df[cust_col] == selected_cust]
        
        # CIE Color Filter
        cie_options = sorted(base_df[cie_col].unique().astype(str))
        selected_cies = st.multiselect("3. Select CIE Color Codes for Breakdown:", cie_options, default=cie_options[:3])

        if selected_cies:
            all_forecasts = []
            
            # Loop for individual CIE Code forecasting
            for cie in selected_cies:
                cie_data = base_df[base_df[cie_col].astype(str) == cie].copy()
                cie_data['month_ds'] = cie_data['ds'].dt.to_period('M').dt.to_timestamp()
                
                # Aggregate by month for Prophet
                p_df = cie_data.groupby('month_ds')['Order qty.(A)'].sum().reset_index().rename(columns={'month_ds':'ds', 'Order qty.(A)':'y'})
                
                if len(p_df) >= 2:
                    m = Prophet(yearly_seasonality=True, daily_seasonality=False).fit(p_df)
                    future = m.make_future_dataframe(periods=12, freq='MS')
                    forecast = m.predict(future)
                    
                    # Filter for 2026 and clean results
                    f2026 = forecast[forecast['ds'].dt.year == 2026][['ds', 'yhat']].copy()
                    f2026['yhat'] = f2026['yhat'].clip(lower=0)
                    f2
