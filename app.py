import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

# 1. Page Configuration
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
    
    # 2. Sidebar Settings
    st.sidebar.header("Data Configuration")
    cust_col = st.sidebar.selectbox("Customer ID/Name Column:", df.columns)
    cie_col = st.sidebar.selectbox("CIE Color Column:", df.columns)
    
    # Select Customer first to trigger specific Pareto logic
    cust_list = sorted(df[cust_col].unique().astype(str))
    selected_cust = st.selectbox("1. Select Customer to Analyze:", cust_list)

    if selected_cust:
        # FILTER DATA FOR SELECTED CUSTOMER
        cust_df = df[df[cust_col].astype(str) == selected_cust].copy()
        
        # --- PER-CUSTOMER PARETO ANALYSIS ---
        sales_summary = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        sales_summary['Cum_Pct'] = sales_summary['M USD'].cumsum() / sales_summary['M USD'].sum()
        
        # Identify top products for THIS customer
        top_products = sales_summary[sales_summary['Cum_Pct'] <= 0.81]['Material name'].unique()
        
        st.subheader(f"📊 Pareto Analysis for {selected_cust}")
        st.info(f"Top products contributing to ~80% of revenue: {len(top_products)}")

        selected_prod = st.selectbox("2. Select Top Product:", top_products)

        if selected_prod:
            prod_df = cust_df[cust_df['Material name'] == selected_prod].copy()
            cie_options = sorted(prod_df[cie_col].unique().astype(str))
            selected_cies = st.multiselect("3. Select CIE Color Codes:", cie_options, default=cie_options[:1])

            if selected_cies:
                results = []
                for cie in selected_cies:
                    cie_df = prod_df[prod_df[cie_col].astype(str) == cie].copy()
                    cie_df['m'] = cie_df['ds'].dt.to_period('M').dt.to_timestamp()
                    p_df = cie_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
                    
                    if len(p_df) >= 2:
                        m = Prophet(yearly_seasonality=True).fit(p_df)
                        future = m.make_future_dataframe(periods=12, freq='MS')
                        fcst = m.predict(future)
                        f26 = fcst[fcst['ds'].dt.year == 2026][['ds', 'yhat']].copy()
                        f26['yhat'] = f26['yhat'].clip(lower=0)
                        f26['CIE'] = cie
                        results.append(f26)
                
                if results:
                    final_df = pd.concat(results)
                    final_df['Month'] = final_df['ds'].dt.strftime('%m/%Y')
                    
                    # 4. Display Visuals
                    st.divider()
                    fig = px.line(final_df, x='Month', y='yhat', color='CIE', markers=True, 
                                  title=f"2026 Forecast: {selected_prod} for {selected_cust}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Forecast Data Breakdown")
                    table_df = final_df[['Month', 'CIE', 'yhat']].rename(columns={'yhat': 'Qty (Pcs)'})
                    st.dataframe(table_df.style.format("{:,.0f}", subset=['Qty (Pcs)']), use_container_width=True)
                else:
                    st.warning("Insufficient data for AI forecast on these codes.")
            else:
                st.info("Please select at least one CIE code.")

except Exception as e:
    st.error(f"Execution Error: {e}")
