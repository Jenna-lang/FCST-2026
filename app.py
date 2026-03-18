import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

# Basic Page Configuration
st.set_page_config(page_title="2026 FCST & CIE Analysis", layout="wide")
st.title("🚀 Supply Chain Forecasting & CIE Color Analysis 2026")

@st.cache_data
def load_data():
    # Loading data and cleaning column names
    df = pd.read_excel('AICheck.xlsx')
    df.columns = [str(col).strip() for col in df.columns]
    df['ds'] = pd.to_datetime(df['Requested deliv. date'])
    return df

try:
    df = load_data()
    
    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.header("⚙️ Data Settings")
    # Mapping columns to avoid showing raw IDs like 700153
    cust_col = st.sidebar.selectbox("Customer Name Column:", df.columns)
    cie_col = st.sidebar.selectbox("CIE Color Code Column:", df.columns)
    
    # Material Selection
    prod_list = sorted(df['Material name'].unique())
    selected_prod = st.sidebar.selectbox("Select Product:", prod_list)

    if selected_prod:
        prod_df = df[df['Material name'] == selected_prod].copy()
        
        # Customer Selection
        cust_list = sorted([str(x) for x in prod_df[cust_col].unique()])
        selected_cust = st.selectbox("Select Customer:", ["ALL Customers"] + cust_list)
        
        w_df = prod_df if selected_cust == "ALL Customers" else prod_df[prod_df[cust_col] == selected_cust]
        
        # CIE Color Code Selection
        cie_options = sorted(w_df[cie_col].unique().astype(str))
        selected_cies = st.multiselect("Select CIE Color Codes for Detailed Forecast:", cie_options, default=cie_options[:3])

        if selected_cies:
            all_results = []
            
            # AI Forecasting Loop for each CIE code
            for cie in selected_cies:
                cie_data = w_df[w_df[cie_col].astype(str) == cie].copy()
                cie_data['ds_m'] = cie_data['ds'].dt.to_period('M').dt.to_timestamp()
                p_df = cie_data.groupby('ds_m')['Order qty.(A)'].sum().reset_index().rename(columns={'ds_m':'ds', 'Order qty.(A)':'y'})
                
                if len(p_df) >= 2:
                    m = Prophet(yearly_seasonality=True).fit(p_df)
                    future = m.make_future_dataframe(periods=12, freq='MS')
                    fcst = m.predict(future)
                    
                    # Filtering for 2026 results
                    f2026 = fcst[fcst['ds'].dt.year == 2026][['ds', 'yhat']].copy()
                    f2026['yhat'] = f2026['yhat'].clip(lower=0)
                    f2026['CIE Code'] = cie
                    f2026['Product Name'] = selected_prod
                    all_results.append(f2026)
            
            if all_results:
