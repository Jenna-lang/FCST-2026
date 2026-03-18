import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

# 1. Page Setup
st.set_page_config(page_title="Customer Pareto Forecast", layout="wide")
st.title("🎯 Customer-Specific Pareto & CIE Analysis 2026")

@st.cache_data
def load_data():
    df = pd.read_excel('AICheck.xlsx')
    df.columns = [str(col).strip() for col in df.columns]
    df['ds'] = pd.to_datetime(df['Requested deliv. date'])
    return df

try:
    df = load_data()
    
    # 2. Sidebar Configuration
    st.sidebar.header("Settings")
    cust_col = st.sidebar.selectbox("Customer Name Column:", df.columns)
    cie_col = st.sidebar.selectbox("CIE Color Column:", df.columns)
    
    # Get Unique Customer List
    cust_list = sorted(df[cust_col].unique().astype(str))
    selected_cust = st.selectbox("1. Select Customer to Analyze:", cust_list)

    if selected_cust:
        # FILTER DATA BY SELECTED CUSTOMER FIRST
        cust_df = df[df[cust_col].astype(str) == selected_cust].copy()
        
        # --- PER-CUSTOMER PARETO ANALYSIS (80/20) ---
        # Group by material for THIS customer specifically
        pareto_df = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        pareto_df['Cumulative_Sales'] = pareto_df['M USD'].cumsum()
        total_sales = pareto_df['M USD'].sum()
        pareto_df['Sales_Pct'] = pareto_df['Cumulative_Sales'] / total_sales
        
        # Identify products contributing to top 80% revenue for this customer
        top_products = pareto_df[pareto_df['Sales_Pct'] <= 0.81]['Material name'].unique()
        
        st.subheader(f"📊 Pareto Analysis for {selected_cust}")
        st.write(f"Found **{len(top_products)}** products contributing to ~80% of this customer's revenue.")

        # 3. Product & CIE Selection
        selected_prod = st.selectbox("2. Select Top Product:", top_products)

        if selected_prod:
            prod_df = cust_df[cust_df['Material name'] == selected_prod].copy()
            cie_options = sorted(prod_df[cie_col].unique().astype(str))
            selected_cies = st.multiselect("3. Select CIE Color Codes:", cie_options, default=cie_options[:1])

            if selected_cies:
                all_forecasts = []
                
                # Forecast loop
                for cie in selected_cies:
                    cie_df = prod_df[prod_df[cie_col].astype(str) == cie].copy()
                    cie_df['m_ds'] = cie_df['ds'].dt.to_period('M').dt.to_timestamp()
                    p_df = cie_df.groupby('m_ds')['Order qty.(A)'].sum().reset_index().rename(columns={'m_ds':'ds', 'Order qty.(A)':'y'})
                    
                    if len(p_df) >= 2:
                        m = Prophet(yearly_seasonality=True).fit(p_df)
                        future = m.make_future_dataframe(periods=12, freq='MS')
                        fcst = m.predict(future)
                        
                        f26 = fcst[fcst['ds'].dt.year == 2026][['ds', 'yhat']].copy()
                        f26['yhat'] = f26['yhat'].clip(lower=0)
                        f26['CIE Code'] = cie
                        f26['Product'] = selected_prod
                        all_forecasts.append(f26)
                
                if all_forecasts:
                    final_df = pd.concat(all_forecasts)
                    final_df['Month'] = final_df['ds'].dt.strftime('%m/%Y')
                    
                    # 4. Display Visuals
                    st.divider()
                    fig = px.line(final_df, x='Month', y='yhat', color='CIE Code', markers=True, 
                                  title=f"2026 Forecast Trend: {selected_prod} for {selected_cust}")
