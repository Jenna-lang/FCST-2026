import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="AI Supply Chain Advisor", layout="wide")
st.title("🤖 AI Strategic Advisor: Product-Level Analysis")

@st.cache_data
def load_data():
    try:
        df = pd.read_excel('AICheck.xlsx')
        df.columns = [str(col).strip() for col in df.columns]
        df['ds'] = pd.to_datetime(df['Requested deliv. date'])
        return df
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return None

df = load_data()

if df is not None:
    # 2. Sidebar Settings
    st.sidebar.header("Configuration")
    cust_col = st.sidebar.selectbox("Customer Name Column:", df.columns)
    cie_col = st.sidebar.selectbox("CIE Color Column:", df.columns)
    
    cust_list = sorted(df[cust_col].unique().astype(str))
    selected_cust = st.sidebar.selectbox("1. Select Customer:", cust_list)

    if selected_cust:
        cust_df = df[df[cust_col].astype(str) == selected_cust].copy()
        
        # --- PARETO 80/20 BY CUSTOMER ---
        sales = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        sales['Cum_Pct'] = sales['M USD'].cumsum() / sales['M USD'].sum()
        top_prods = sales[sales['Cum_Pct'] <= 0.81]['Material name'].unique()
        
        selected_prod_chart = st.selectbox("2. Select Product for Analysis:", top_prods)
        
        all_pareto_forecasts = []
        product_chart_data = None
        ai_insights = ""

        # --- AI PROCESSING ---
        with st.spinner('Calculating Forecasts...'):
            # A. Logic cho Biểu đồ: Gộp tất cả CIE của sản phẩm đang chọn
            prod_chart_df = cust_df[cust_df['Material name'] == selected_prod_chart].copy()
            prod_chart_df['m'] = prod_chart_df['ds'].dt.to_period('M').dt.to_timestamp()
            # Group by month only (combining all CIEs)
            actual_prod = prod_chart_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
            
            if len(actual_prod) >= 2:
                m_prod = Prophet(yearly_seasonality=True).fit(actual_prod)
                future_prod = m_prod.make_future_dataframe(periods=12, freq='MS')
                fcst_prod = m_prod.predict(future_prod)
                fcst_prod['yhat'] = fcst_prod['yhat'].clip(lower=0)
                product_chart_data = {'actual': actual_prod, 'fcst': fcst_prod}
                
                # AI Insight for the whole product
                avg = actual_prod['y'].tail(12).mean()
                f_2026_max = fcst_prod[fcst_prod['ds'].dt.year == 2026]['yhat'].max()
                growth = ((f_2026_max - avg) / avg * 100) if avg > 0 else 0
                ai_insights = f"**{selected_prod_chart}**: Overall trend is **{'Increasing' if growth > 5 else 'Decreasing' if growth < -5 else 'Stable'}** ({growth:.1f}% vs last 12m average)."

            # B. Logic cho Bảng: Vẫn giữ chi tiết từng CIE của toàn bộ Pareto
            for prod in top_prods:
                temp_df = cust_df[cust_df['Material name'] == prod].copy()
                for cie in temp_df[cie_col].unique():
                    cie_df = temp_df[temp_df[cie_col].astype(str) == cie].copy()
                    cie_df['m'] = cie_df['ds'].dt.to_period('M').dt.to_timestamp()
                    actual_cie = cie_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
                    
                    if len(actual_cie) >= 2:
                        m_cie = Prophet(yearly_seasonality=True).fit(actual_cie)
                        fcst_cie = m_cie.predict(m_cie.make_future_dataframe(periods=12, freq='MS'))
                        f26 = fcst_cie[fcst_cie['ds'].dt.year == 2026].copy()
                        f26['Month'] = f26['ds'].dt.strftime('%m/%Y')
                        f26['Product'] = prod
                        f26['CIE'] = cie
                        all_pareto_forecasts.append(f26[['Month', 'Product', 'CIE', 'yhat']])

        # --- UI DISPLAY ---
        tab1, tab2 = st.tabs(["📊 Product Analysis Chart", "📋 All Pareto FCST Details"])

        with tab1:
            if product_chart_data:
                st.subheader(f"Total Demand Trend: {selected_prod_chart}")
                st.info(ai_insights)
                fig = go.Figure()
                # Actual (Solid line)
                fig.add_trace(go.Scatter(x=product_chart_data['actual']['ds'], y=product_chart_data['actual']['y'], 
                                         mode='lines+markers', name="Total Actual Sales"))
                # Forecast (Dashed line)
                f_only = product_chart_data['fcst'][product_chart_data['fcst']['ds'] > product_chart_data['actual']['ds'].max()]
                fig.add_trace(go.Scatter(x=f_only['ds'], y=f_only['yhat'], 
                                         line=dict(dash='dash', width=4), name="Total 2026 Forecast"))
                
                fig.update_layout(xaxis_title="Month", yaxis_title="Total Qty (Pcs)", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if all_pareto_forecasts:
                st.subheader(f"Full Pareto Breakdown for {selected_cust}")
                final_df = pd.concat(all_pareto_forecasts)
                st.dataframe(final_df.rename(columns={'yhat': 'Qty (Pcs)'}).style.format("{:,.0f}", subset=['Qty (Pcs)']), use_container_width=True)
                
                csv = final_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 Download Full Report", data=csv, file_name=f"Pareto_FCST_{selected_cust}.csv")
