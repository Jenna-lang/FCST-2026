import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="AI Supply Chain Advisor", layout="wide")
st.title("🤖 AI Strategic Advisor: Year-over-Year Comparison")

@st.cache_data
def load_data():
    try:
        df = pd.read_excel('AICheck.xlsx')
        df.columns = [str(col).strip() for col in df.columns]
        df['ds'] = pd.to_datetime(df['Requested deliv. date'])
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

df = load_data()

if df is not None:
    # 2. Sidebar Settings
    cust_col = st.sidebar.selectbox("Customer Column:", df.columns)
    cie_col = st.sidebar.selectbox("CIE Column:", df.columns)
    
    cust_list = sorted(df[cust_col].unique().astype(str))
    selected_cust = st.sidebar.selectbox("1. Select Customer:", cust_list)

    if selected_cust:
        cust_df = df[df[cust_col].astype(str) == selected_cust].copy()
        
        # Pareto 80/20 Logic
        sales = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        sales['Cum_Pct'] = sales['M USD'].cumsum() / sales['M USD'].sum()
        top_prods = sales[sales['Cum_Pct'] <= 0.81]['Material name'].unique()
        
        selected_prod = st.selectbox("2. Select Product for Analysis:", top_prods)
        
        all_results = []
        chart_data = None
        comp_data = None
        ai_insights = ""

        with st.spinner('Analyzing Yearly Growth...'):
            # A. Biểu đồ đường & So sánh tổng sản lượng
            prod_df = cust_df[cust_df['Material name'] == selected_prod].copy()
            prod_df['m'] = prod_df['ds'].dt.to_period('M').dt.to_timestamp()
            actual_prod = prod_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
            
            if len(actual_prod) >= 2:
                m_prod = Prophet(yearly_seasonality=True).fit(actual_prod)
                fcst_prod = m_prod.predict(m_prod.make_future_dataframe(periods=12, freq='MS'))
                fcst_prod['yhat'] = fcst_prod['yhat'].clip(lower=0)
                chart_data = {'actual': actual_prod, 'fcst': fcst_prod}
                
                # --- LOGIC: SO SÁNH TỔNG SẢN LƯỢNG NĂM ---
                sum_2025 = actual_prod[actual_prod['ds'].dt.year == 2025]['y'].sum()
                sum_2026_fcst = fcst_prod[fcst_prod['ds'].dt.year == 2026]['yhat'].sum()
                
                comp_data = {'labels': ['2025 (Actual)', '2026 (Forecast)'], 'values': [sum_2025, sum_2026_fcst]}
                
                if sum_2025 > 0:
                    growth = ((sum_2026_fcst - sum_2025) / sum_2025) * 100
                    status = "Increasing" if growth > 10 else "Decreasing" if growth < -10 else "Stable"
                    ai_insights = f"**{selected_prod}**: Total yearly demand is projected to be **{status}** at **{growth:.1f}%** (Avg vs Avg)."
                else:
                    ai_insights = f"**{selected_prod}**: New product entry (Insufficient 2025 data)."

            # B. Bảng chi tiết cho toàn bộ Pareto
            for p in top_prods:
                p_df = cust_df[cust_df['Material name'] == p].copy()
                for c in p_df[cie_col].unique():
                    c_df = p_df[p_df[cie_col].astype(str) == c].copy()
                    c_df['m'] = c_df['ds'].dt.to_period('M').dt.to_timestamp()
                    act_c = c_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
                    if len(act_c) >= 2:
                        m_c = Prophet().fit(act_c)
                        f_c = m_c.predict(m_c.make_future_dataframe(periods=12, freq='MS'))
                        res26 = f_c[f_c['ds'].dt.year == 2026].copy()
                        res26['Month'] = res26['ds'].dt.strftime('%m/%Y'); res26['Product'] = p; res26['CIE'] = c
                        all_results.append(res26[['Month', 'Product', 'CIE', 'yhat']])

        # --- Giao diện Tabs ---
        tab1, tab2 = st.tabs(["📊 Yearly Comparison", "📋 Full Pareto Details"])

        with tab1:
            if chart_data and comp_data:
                st.info(ai_insights)
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**Monthly Trend**")
                    fig_line = go.Figure()
                    fig_line.add_trace(go.Scatter(x=chart_data['actual']['ds'], y=chart_data['actual']['y'], name="Actual"))
                    f_only = chart_data['fcst'][chart_data['fcst']['ds'] > chart_data['actual']['ds'].max()]
                    fig_line.add_trace(go.Scatter(x=f_only['ds'], y=f_only['yhat'], line=dict(dash='dash'), name="2026 FCST"))
                    st.plotly_chart(fig_line, use_container_width=True)
                
                with col2:
                    st.write("**Yearly Volume**")
                    fig_bar = go.Figure(data=[go.Bar(x=comp_data['labels'], y=comp_data['values'], marker_color=['#1f77b4', '#ff7f0e'])])
                    fig_bar.update_layout(height=400)
                    st.plotly_chart(fig_bar, use_container_width=True)

        with tab2:
            if all_results:
                final_table = pd.concat(all_results).rename(columns={'yhat': 'Qty (Pcs)'})
                st.dataframe(final_table.style.format("{:,.0f}", subset=['Qty (Pcs)']), use_container_width=True)
