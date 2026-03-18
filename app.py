import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="2026 Forecast Analysis", layout="wide")
st.title("🚀 Supply Chain Forecast & CIE Analysis 2026")

@st.cache_data
def load_data():
    # Loading the Excel file
    df = pd.read_excel('AICheck.xlsx')
    df.columns = [str(col).strip() for col in df.columns]
    df['ds'] = pd.to_datetime(df['Requested deliv. date'])
    return df

try:
    df = load_data()
    
    # 2. Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    cust_col = st.sidebar.selectbox("Customer Name Column:", df.columns)
    cie_col = st.sidebar.selectbox("CIE Color Code Column:", df.columns)
    
    # Pareto Filter (Top 85%)
    summary = df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
    summary['Cum_Pct'] = summary['M USD'].cumsum() / summary['M USD'].sum()
    pareto_list = summary[summary['Cum_Pct'] <= 0.85]['Material name'].unique()

    selected_prod = st.sidebar.selectbox("Select Product:", pareto_list)

    if selected_prod:
        prod_df = df[df['Material name'] == selected_prod].copy()
        
        st.subheader(f"🔍 Product Analysis: {selected_prod}")
        
        # 3. Customer & CIE Filters
        cust_list = sorted([str(x) for x in prod_df[cust_col].unique()])
        selected_cust = st.selectbox("Select Customer:", ["ALL Customers"] + cust_list)
        
        # Filter data based on customer
        base_df = prod_df if selected_cust == "ALL Customers" else prod_df[prod_df[cust_col] == selected_cust]
        
        cie_options = sorted(base_df[cie_col].unique().astype(str))
        selected_cies = st.multiselect("Select CIE Color Codes:", cie_options, default=cie_options[:1])

        if selected_cies:
            all_forecasts = []
            
            # AI Forecasting Loop
            for cie in selected_cies:
                cie_data = base_df[base_df[cie_col].astype(str) == cie].copy()
                cie_data['month_ds'] = cie_data['ds'].dt.to_period('M').dt.to_timestamp()
                p_df = cie_data.groupby('month_ds')['Order qty.(A)'].sum().reset_index().rename(columns={'month_ds':'ds', 'Order qty.(A)':'y'})
                
                if len(p_df) >= 2:
                    m = Prophet(yearly_seasonality=True).fit(p_df)
                    future = m.make_future_dataframe(periods=12, freq='MS')
                    forecast = m.predict(future)
                    
                    # 2026 Results
                    f2026 = forecast[forecast['ds'].dt.year == 2026][['ds', 'yhat']].copy()
                    f2026['
