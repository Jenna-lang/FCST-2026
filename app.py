import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

# 1. Page Setup
st.set_page_config(page_title="FCST 2026", layout="wide")
st.title("🚀 Supply Chain Forecast & CIE Color Analysis 2026")

@st.cache_data
def load_data():
    df = pd.read_excel('AICheck.xlsx')
    df.columns = [str(col).strip() for col in df.columns]
    df['ds'] = pd.to_datetime(df['Requested deliv. date'])
    return df

try:
    df = load_data()
    
    # 2. Settings Sidebar
    st.sidebar.header("Settings")
    cust_col = st.sidebar.selectbox("Select Customer Column:", df.columns)
    cie_col = st.sidebar.selectbox("Select CIE Color Column:", df.columns)
    
    # Filter for top products
    prod_list = sorted(df['Material name'].unique())
    selected_prod = st.sidebar.selectbox("Select Product:", prod_list)

    if selected_prod:
        prod_df = df[df['Material name'] == selected_prod].copy()
        
        # 3. Filters
        cust_list = sorted([str(x) for x in prod_df[cust_col].unique()])
        selected_cust = st.selectbox("Select Customer:", ["ALL"] + cust_list)
        
        w_df = prod_df if selected_cust == "ALL" else prod_df[prod_df[cust_col] == selected_cust]
        
        cie_options = sorted(w_df[cie_col].unique().astype(str))
        selected_cies = st.multiselect("Select CIE Color Codes:", cie_options, default=cie_options[:1])

        if selected_cies:
            results = []
            
            # Forecast loop for each CIE code
            for cie in selected_cies:
                cie_df = w_df[w_df[cie_col].astype(str) == cie].copy()
                cie_df['m'] = cie_df['ds'].dt.to_period('M').dt.to_timestamp()
                p_df = cie_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
                
                if len(p_df) >= 2:
                    m = Prophet(yearly_seasonality=True).fit(p_df)
                    future = m.make_future_dataframe(periods=12, freq='MS')
                    fcst = m.predict(future)
                    
                    f26 = fcst[fcst['ds'].dt.year == 2026][['ds', 'yhat']].copy()
                    f26['yhat'] = f26['yhat'].clip(lower=0)
                    f26['CIE Code'] = cie
                    f26['Product'] = selected_prod
                    results.append(f26)
            
            if results:
                final_df = pd.concat(results)
                final_df['Month'] = final_df['ds'].dt.strftime('%m/%Y')
                
                # 4. Display Results
                st.divider()
                fig = px.line(final_df, x='Month', y='yhat', color='CIE Code', markers=True, title="2026 Trend")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Forecast Data Table")
                out_df = final_df[['Month', 'Product', 'CIE Code', 'yhat']]
                out_df.columns = ['Month', 'Product', 'CIE Code', 'Quantity (Pcs)']
                st.dataframe(out_df.style.format("{:,.0f}", subset=['Quantity (Pcs)']), use_container_width=True)
            else:
                st.warning("Not enough data for a forecast.")
        else:
            st.info("Please select a CIE code.")

except Exception as e:
    st.error(f"Error: {e}")
