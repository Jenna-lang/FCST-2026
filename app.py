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
        df['ds'] = pd.to_datetime(df['Requested deliv. date'], errors='coerce')
        df = df.dropna(subset=['ds'])
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

@st.cache_resource
def fast_forecast(data_series):
    if len(data_series) < 2: return None
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(data_series)
    future = m.make_future_dataframe(periods=12, freq='MS')
    fcst = m.predict(future)
    return fcst[['ds', 'yhat']]

df = load_data()

if df is not None:
    # 2. Sidebar Settings
    st.sidebar.header("⚡ Configuration")
    cust_col = st.sidebar.selectbox("Customer Column:", [c for c in df.columns if 'Customer' in c] or df.columns)
    cie_col = st.sidebar.selectbox("CIE/Color Column:", [c for c in df.columns if 'CIE' in c] or df.columns)
    
    cust_list = sorted(df[cust_col].unique().astype(str))
    selected_cust = st.sidebar.selectbox("1. Select Customer:", ["-- Select --"] + cust_list)

    if selected_cust != "-- Select --":
        cust_df = df[df[cust_col].astype(str) == selected_cust].copy()
        
        # Pareto 80/20 Analysis
        sales = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        sales['Cum_Pct'] = sales['M USD'].cumsum() / (sales['M USD'].sum() or 1)
        top_prods = sales[sales['Cum_Pct'] <= 0.81]['Material name'].unique()
        
        selected_prod = st.selectbox("2. Select Product for Analysis:", top_prods if len(top_prods)>0 else df['Material name'].unique()[:5])
        
        growth_final = 0.0 

        tab1, tab2 = st.tabs(["📊 Growth Analysis", "📋 2026 Production Plan"])

        # --- TAB 1: GROWTH CALCULATION ---
        with tab1:
            prod_df = cust_df[cust_df['Material name'] == selected_prod].copy()
            prod_df['m'] = prod_df['ds'].dt.to_period('M').dt.to_timestamp()
            actual_all = prod_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
            
            act_2026 = actual_all[actual_all['ds'].dt.year == 2026]
            last_act_date = act_2026['ds'].max() if not act_2026.empty else pd.Timestamp('2025-12-31')
            
            res = fast_forecast(actual_all)
            if res is not None:
                total_25 = actual_all[actual_all['ds'].dt.year == 2025]['y'].sum()
                sum_act_26 = act_2026['y'].sum()
                fcst_future_26 = res[(res['ds'].dt.year == 2026) & (res['ds'] > last_act_date)]['yhat'].sum()
                total_26_mixed = sum_act_26 + fcst_future_26
                
                growth_final = ((total_26_mixed - total_25) / total_25 * 100) if total_25 > 0 else 0
                st.metric("Projected 2026 Growth Rate", f"{growth_final:.1f}%")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=actual_all['ds'], y=actual_all['y'], name="Actual", line=dict(color='#1f77b4')))
                fig.add_trace(go.Scatter(x=res[res['ds'] > last_act_date]['ds'], y=res[res['ds'] > last_act_date]['yhat'], 
                                         line=dict(dash='dash', color='#ff7f0e'), name="Forecast"))
                st.plotly_chart(fig, use_container_width=True)

        # --- TAB 2: PRODUCTION PLAN (FIXED NO-DATA ERROR) ---
        with tab2:
            st.subheader("📋 2026 Detailed Supply Plan")
            
            # 1. Prepare Maps
            # Historical 2025 seasonality
            df_25 = cust_df[cust_df['ds'].dt.year == 2025]
            act_25_map = df_25.groupby(['Material name', cie_col, df_25['ds'].dt.month])['Order qty.(A)'].sum().to_dict()
            
            # Actual 2026 (T1-T3)
            df_26 = cust_df[cust_df['ds'].dt.year == 2026]
            act_26_map = df_26.groupby(['Material name', cie_col, df_26['ds'].dt.month])['Order qty.(A)'].sum().to_dict()
            
            # Run-rate (T1-T3 average)
            run_rate_map = df_26.groupby(['Material name', cie_col])['Order qty.(A)'].mean().to_dict()

            g_factor = 1 + (growth_final / 100)
            months_26 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
            cols_26 = [m.strftime('%m/%Y') for m in months_26]
            
            # FIX: Get ALL items from BOTH years
            all_items = cust_df[cust_df['Material name'].isin(top_prods)][['Material name', cie_col]].drop_duplicates()
            
            pivot_list = []
            for _, r in all_items.iterrows():
                row = {'Product': r['Material name'], 'CIE': r[cie_col]}
                p, c = r['Material name'], r[cie_col]
                
                for m_date in months_26:
                    m_idx = m_date.month
                    m_str = m_date.strftime('%m/%Y')
                    
                    # PRIORITY 1: Actuals 2026
                    if (p, c, m_idx) in act_26_map:
                        row[m_str] = act_26_map[(p, c, m_idx)]
                    # PRIORITY 2: Seasonality 2025 * Growth
                    elif (p, c, m_idx) in act_25_map:
                        row[m_str] = round(act_25_map[(p, c, m_idx)] * g_factor, 0)
                    # PRIORITY 3: Run-rate 2026 (For New Projects) * Growth
                    elif (p, c) in run_rate_map:
                        row[m_str] = round(run_rate_map[(p, c)] * g_factor, 0)
                    else:
                        row[m_str] = 0
                pivot_list.append(row)
            
            res_df = pd.DataFrame(pivot_list)
            
            if not res_df.empty:
                st.info(f"📌 Applied Growth Factor: {g_factor:.2f}x (Based on {selected_prod})")
                st.dataframe(res_df.style.format("{:,.0f}", subset=cols_26), use_container_width=True)
                st.download_button("📥 Download Plan (CSV)", res_df.to_csv(index=False).encode('utf-8-sig'), "Supply_Plan_2026.csv")
            else:
                st.warning("No data found. Please check your Customer/CIE selection.")
