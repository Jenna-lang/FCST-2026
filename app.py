import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. System Configuration
st.set_page_config(page_title="AI Supply Chain Advisor 2026", layout="wide")

@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_excel('AICheck.xlsx')
        df.columns = [str(col).strip() for col in df.columns]
        for col in df.select_dtypes(['object']).columns:
            df[col] = df[col].astype(str).str.strip()
        df['ds'] = pd.to_datetime(df['Requested deliv. date'], errors='coerce')
        df = df.dropna(subset=['ds'])
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

@st.cache_data(ttl=3600)
def get_detailed_fcst(cust_df, prod_name):
    p_df = cust_df[cust_df['Material name'] == prod_name].copy()
    if len(p_df) < 5: return None, 0.0
    
    p_df['m'] = p_df['ds'].dt.to_period('M').dt.to_timestamp()
    df_p = p_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
    
    try:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(df_p)
        future = m.make_future_dataframe(periods=12, freq='MS')
        fcst = m.predict(future)
        
        t25 = df_p[df_p['ds'].dt.year == 2025]['y'].sum()
        a26 = df_p[df_p['ds'].dt.year == 2026]['y'].sum()
        f26 = fcst[(fcst['ds'].dt.year == 2026) & (fcst['ds'] > df_p['ds'].max())]['yhat'].sum()
        growth = ((a26 + f26 - t25) / t25) if t25 > 0 else 0.0
        return fcst[['ds', 'yhat']], growth
    except:
        return None, 0.0

df = load_data()

if df is not None:
    st.sidebar.header("⚡ Settings")
    cust_col = st.sidebar.selectbox("Customer:", [c for c in df.columns if 'Customer' in c] or [df.columns[0]])
    cie_col = st.sidebar.selectbox("CIE/Color:", [c for c in df.columns if 'CIE' in c] or [df.columns[1]])
    adj_growth = st.sidebar.slider("AI Adjustment (%)", -50, 50, 0)
    
    cust_list = sorted(df[cust_col].unique())
    selected_cust = st.sidebar.selectbox("1. Select Customer:", ["-- Select --"] + cust_list)

    if selected_cust != "-- Select --":
        cust_df = df[df[cust_col] == selected_cust].copy()
        sales = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        sales['Cum_Pct'] = sales['M USD'].cumsum() / (sales['M USD'].sum() or 1)
        top_prods = sales[sales['Cum_Pct'] <= 0.85]['Material name'].unique()
        
        tab1, tab2 = st.tabs(["📊 Trend & Comparison", "📋 2026 Production Plan"])

        with tab1:
            selected_prod = st.selectbox("Select Product to Analyze:", top_prods)
            target_annual = st.sidebar.number_input(f"Annual Target for {selected_prod}:", value=100000)
            
            fcst_res, ai_g = get_detailed_fcst(cust_df, selected_prod)
            if fcst_res is not None:
                p_df = cust_df[cust_df['Material name'] == selected_prod].copy()
                p_df['m'] = p_df['ds'].dt.to_period('M').dt.to_timestamp()
                actuals = p_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=actuals[actuals['ds'].dt.year < 2026]['ds'], y=actuals[actuals['ds'].dt.year < 2026]['y'], name="History", line=dict(color='gray', width=1)))
                fig.add_trace(go.Scatter(x=actuals[actuals['ds'].dt.year == 2026]['ds'], y=actuals[actuals['ds'].dt.year == 2026]['y'], name="Actual 2026", mode='lines+markers', line=dict(color='blue', width=3)))
                fig.add_trace(go.Scatter(x=fcst_res[fcst_res['ds'] >= '2026-01-01']['ds'], y=fcst_res[fcst_res['ds'] >= '2026-01-01']['yhat'], name="AI Forecast", line=dict(color='orange', dash='dash')))
                fig.add_trace(go.Scatter(x=pd.date_range('2026-01-01', '2026-12-01', freq='MS'), y=[target_annual/12]*12, name="Target", line=dict(color='green', dash='dot')))
                
                fig.update_layout(title=f"Analysis: {selected_prod}", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("📋 2026 Supply Plan")
            with st.spinner("Calculating..."):
                growth_map = {p: get_detailed_fcst(cust_df, p)[1] for p in top_prods}
            
            df_25 = cust_df[cust_df['ds'].dt.year == 2025]
            df_26 = cust_df[cust_df['ds'].dt.year == 2026]
            act_25_map = df_25.groupby(['Material name', cie_col, df_25['ds'].dt.month])['Order qty.(A)'].sum().to_dict()
            act_26_map = df_26.groupby(['Material name', cie_col, df_26['ds'].dt.month])['Order qty.(A)'].sum().to_dict()
            run_rate_map = df_26[df_26['ds'].dt.month <= 3].groupby(['Material name', cie_col])['Order qty.(A)'].mean().to_dict()

            months_26 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
            cols_26 = [m.strftime('%m/%Y') for m in months_26]
            
            all_items = cust_df[cust_df['Material name'].isin(top_prods)][['Material name', cie_col]].drop_duplicates()
            pivot_list = []
            for _, r in all_items.iterrows():
                p, c = r['Material name'], r[cie_col]
                factor = 1 + growth_map.get(p, 0.0) + (adj_growth / 100)
                row = {'Product': p, 'CIE': c, 'Growth': f"{(factor-1)*100:.1f}%"}
                for m_date in months_26:
                    m_idx, m_str = m_date.month, m_date.strftime('%m/%Y')
                    if (p, c, m_idx) in act_26_map: row[m_str] = act_26_map[(p, c, m_idx)]
                    elif m_idx > 3:
                        if (p, c, m_idx) in act_25_map: row[m_str] = round(act_25_map[(p, c, m_idx)] * factor, 0)
                        elif (p, c) in run_rate_map: row[m_str] = round(run_rate_map[(p, c)] * factor, 0)
                        else: row[m_str] = 0
                    else: row[m_str] = 0
                pivot_list.append(row)
            
            if pivot_list:
                res_df = pd.DataFrame(pivot_list)
                # Dòng tổng cộng an toàn
                total_row = {'Product': 'GRAND TOTAL', 'CIE': '---', 'Growth': '---'}
                for col in cols_26:
                    total_row[col] = res_df[col].sum()
                res_df = pd.concat([res_df, pd.DataFrame([total_row])], ignore_index=True)
                
                st.dataframe(res_df.style.apply(lambda x: ['font-weight:bold; background:#e6f3ff' if x['Product']=='GRAND TOTAL' else '' for _ in x], axis=1).format("{:,.0f}", subset=cols_26), use_container_width=True)
