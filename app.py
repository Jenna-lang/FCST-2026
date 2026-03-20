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
        st.error(f"Error loading Excel file: {e}")
        return None

@st.cache_data(ttl=3600)
def get_comprehensive_analysis(cust_df, prod_name):
    p_df = cust_df[cust_df['Material name'] == prod_name].copy()
    if len(p_df) < 3: return None, None, 0.0
    
    p_df['m'] = p_df['ds'].dt.to_period('M').dt.to_timestamp()
    df_all = p_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
    
    try:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(df_all)
        future = m.make_future_dataframe(periods=12, freq='MS')
        fcst = m.predict(future)
        
        # Calculate AI Growth (Full 2026 Forecast vs 2025 Actual)
        t25 = df_all[df_all['ds'].dt.year == 2025]['y'].sum()
        a26_act = df_all[df_all['ds'].dt.year == 2026]['y'].sum()
        f26_rem = fcst[(fcst['ds'].dt.year == 2026) & (fcst['ds'] > df_all['ds'].max())]['yhat'].sum()
        growth = ((a26_act + f26_rem - t25) / t25) if t25 > 0 else 0.0
        return df_all, fcst[['ds', 'yhat']], growth
    except:
        return None, None, 0.0

df = load_data()

if df is not None:
    st.sidebar.header("⚡ Settings")
    cust_col = st.sidebar.selectbox("Customer Column:", [c for c in df.columns if 'Customer' in c] or [df.columns[0]])
    cie_col = st.sidebar.selectbox("CIE/Color Column:", [c for c in df.columns if 'CIE' in c] or [df.columns[1]])
    adj_growth = st.sidebar.slider("Manual AI Adjustment (%)", -50, 50, 0)
    
    cust_list = sorted(df[cust_col].unique())
    selected_cust = st.sidebar.selectbox("1. Select Customer:", ["-- Select --"] + cust_list)

    if selected_cust != "-- Select --":
        cust_df = df[df[cust_col] == selected_cust].copy()
        
        # --- PARETO 80/20 ANALYSIS ---
        rev_df = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        rev_df['Cum_Pct'] = (rev_df['M USD'].cumsum() / rev_df['M USD'].sum()) * 100
        # Strategic items (top 85% revenue)
        top_prods = rev_df[rev_df['Cum_Pct'] <= 85]['Material name'].unique()[:20]
        
        st.sidebar.success(f"📈 Pareto: Tracking {len(top_prods)} Strategic Items.")

        tab1, tab2 = st.tabs(["📊 Digital Variance & Trends", "📋 2026 Supply Plan"])

        with tab1:
            selected_prod = st.selectbox("Select Strategic Product:", top_prods)
            actuals_all, fcst_res, ai_g = get_comprehensive_analysis(cust_df, selected_prod)
            
            if fcst_res is not None:
                # 1. TREND CHART
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=actuals_all[actuals_all['ds'] < '2026-01-01']['ds'], y=actuals_all[actuals_all['ds'] < '2026-01-01']['y'], name="History (2023-25)", line=dict(color='gray', width=1)))
                fig.add_trace(go.Scatter(x=actuals_all[actuals_all['ds'] >= '2026-01-01']['ds'], y=actuals_all[actuals_all['ds'] >= '2026-01-01']['y'], name="Actual 2026 (Q1)", mode='lines+markers', line=dict(color='blue', width=3)))
                fig.add_trace(go.Scatter(x=fcst_res[fcst_res['ds'] >= '2026-01-01']['ds'], y=fcst_res[fcst_res['ds'] >= '2026-01-01']['yhat'], name="AI Forecast", line=dict(color='orange', dash='dash')))
                fig.update_layout(title=f"Forecasting Trend: {selected_prod}", hovermode="x unified", xaxis_title="Timeline", yaxis_title="Quantity")
                st.plotly_chart(fig, use_container_width=True)

                # 2. DIGITAL VARIANCE TABLE
                st.write("### 🔢 Digital Variance Analysis (Jan - Mar 2026)")
                act_q1 = actuals_all[(actuals_all['ds'] >= '2026-01-01') & (actuals_all['ds'] <= '2026-03-01')]
                fcst_q1 = fcst_res[(fcst_res['ds'] >= '2026-01-01') & (fcst_res['ds'] <= '2026-03-01')]
                
                comp_df = pd.merge(act_q1, fcst_q1, on='ds', how='inner')
                if not comp_df.empty:
                    comp_df['Diff'] = comp_df['y'] - comp_df['yhat']
                    comp_df['Variance %'] = (comp_df['Diff'] / comp_df['yhat']) * 100
                    
                    st.dataframe(comp_df.style.format({
                        'ds': lambda x: x.strftime('%m/%Y'), 'y': '{:,.0f}', 'yhat': '{:,.0f}', 'Diff': '{:,.0f}', 'Variance %': '{:.1f}%'
                    }).applymap(lambda x: 'color: red; font-weight: bold' if abs(x) > 20 else 'color: green', subset=['Variance %']), use_container_width=True)
                else:
                    st.info("No Q1 2026 actual data available for comparison yet.")

        with tab2:
            st.subheader("📋 Supply Plan 2026 (Run-rate Logic for New Items)")
            with st.spinner("Calculating plan for Pareto items..."):
                growth_map = {p: get_comprehensive_analysis(cust_df, p)[2] for p in top_prods}

            months_26 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
            cols_26 = [m.strftime('%m/%Y') for m in months_26]
            all_items = cust_df[cust_df['Material name'].isin(top_prods)][['Material name', cie_col]].drop_duplicates()
            
            pivot_list = []
            for _, r in all_items.iterrows():
                p, c = r['Material name'], r[cie_col]
                factor = 1 + growth_map.get(p, 0.0) + (adj_growth / 100)
                
                # Logic: Run-rate (Avg Jan-Mar 2026)
                act_26_q1 = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & (cust_df['ds'].dt.year==2026) & (cust_df['ds'].dt.month <= 3)]
                run_rate = act_26_q1['Order qty.(A)'].mean() if not act_26_q1.empty else 0
                
                row = {'Product': p, 'CIE': c, 'AI Growth': f"{(factor-1)*100:.1f}%"}
                for m_date in months_26:
                    m_idx, m_str = m_date.month, m_date.strftime('%m/%Y')
                    # 1. Use Actual if exists
                    act_val = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2026)]['Order qty.(A)'].sum()
                    if act_val > 0:
                        row[m_str] = act_val
                    elif m_idx > 3: # Future months
                        # 2. Use 2025 Data
                        h25 = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2025)]['Order qty.(A)'].sum()
                        if h25 > 0:
                            row[m_str] = round(h25 * factor, 0)
                        else:
                            # 3. NEW ITEM LOGIC: Use Run-rate * growth factor
                            row[m_str] = round(run_rate * factor, 0)
                    else:
                        row[m_str] = 0
                pivot_list.append(row)
            
            res_df = pd.DataFrame(pivot_list)
            # Grand Total Row
            total_row = {'Product': 'GRAND TOTAL', 'CIE': '---', 'AI Growth': '---'}
            for col in cols_26: total_row[col] = res_df[col].sum()
            res_df = pd.concat([res_df, pd.DataFrame([total_row])], ignore_index=True)
            
            st.dataframe(res_df.style.apply(lambda x: ['font-weight:bold; background:#e6f3ff' if x['Product']=='GRAND TOTAL' else '' for _ in x], axis=1).format("{:,.0f}", subset=cols_26), use_container_width=True)
            st.download_button("📥 Download Plan (CSV)", res_df.to_csv(index=False).encode('utf-8-sig'), "Supply_Plan_2026.csv")
