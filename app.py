import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. System Configuration
st.set_page_config(page_title="AI Supply Chain Advisor 2026", layout="wide")

# --- DATA PROCESSING ---
def process_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = [str(col).strip() for col in df.columns]
        date_col = 'Requested deliv. date'
        qty_col = 'Order qty.(A)'
        
        if date_col in df.columns and qty_col in df.columns:
            df['ds'] = pd.to_datetime(df[date_col], errors='coerce')
            df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
            df = df.dropna(subset=['ds'])
            return df
        else:
            st.error(f"Required columns '{date_col}' or '{qty_col}' not found!")
            return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def get_metrics(cust_df, prod_name):
    p_df = cust_df[cust_df['Material name'] == prod_name].copy()
    if p_df.empty: return 0.0, 0.0, 0.0
    
    monthly = p_df.groupby(p_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
    monthly['ds_ts'] = monthly['ds'].dt.to_timestamp()
    
    # 1. Avg Growth (Trend across all history 2023-2026)
    monthly['Pct_Change'] = monthly['Order qty.(A)'].pct_change() * 100
    avg_growth = monthly['Pct_Change'].mean() if not monthly['Pct_Change'].empty else 0.0
    
    # 2. 2026 Status
    df_26 = monthly[monthly['ds_ts'].dt.year == 2026]
    act_26_sum = df_26['Order qty.(A)'].sum()
    # Run-rate: Average of actual months in 2026
    run_rate_26 = df_26['Order qty.(A)'].mean() if not df_26.empty else 0.0
    
    # 3. YoY (Comparison for months available in 2026 vs 2025)
    m_26 = df_26['ds_ts'].dt.month.tolist()
    act_25_same = monthly[(monthly['ds_ts'].dt.year == 2025) & (monthly['ds_ts'].dt.month.isin(m_26))]['Order qty.(A)'].sum()
    yoy = (act_26_sum - act_25_same) / act_25_same if act_25_same > 0 else 0.0
    
    return avg_growth, yoy, run_rate_26

# --- SIDEBAR ---
st.sidebar.header("📁 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload AICheck.xlsx", type=['xlsx'])

if uploaded_file:
    df = process_data(uploaded_file)
    if df is not None:
        cust_col = st.sidebar.selectbox("Customer:", [c for c in df.columns if 'Customer' in c] or [df.columns[0]])
        cie_col = st.sidebar.selectbox("CIE/Color:", [c for c in df.columns if 'CIE' in c] or [df.columns[1]])
        adj_growth = st.sidebar.slider("Manual Adjustment (%)", -50, 50, 0)
        selected_cust = st.sidebar.selectbox("Select Customer:", ["-- Select --"] + sorted(df[cust_col].unique()))

        if selected_cust != "-- Select --":
            cust_df = df[df[cust_col] == selected_cust].copy()
            
            # Pareto 85% - focus on top revenue drivers
            rev = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
            rev['Cum_Pct'] = rev['M USD'].cumsum() / rev['M USD'].sum()
            top_prods = rev[rev['Cum_Pct'] <= 0.86]['Material name'].unique()[:20]

            tab1, tab2 = st.tabs(["📊 Performance Audit", "📋 2026 Strategic Plan"])

            with tab1:
                selected_prod = st.selectbox("Product:", top_prods)
                avg_g, yoy_g, r_rate = get_metrics(cust_df, selected_prod)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg Hist Growth", f"{avg_g:.1f}%")
                c2.metric("YoY (26 vs 25)", f"{yoy_g*100:.1f}%")
                c3.metric("2026 Run-rate", f"{r_rate:,.0f}")

                # Charting Logic
                p_plot = cust_df[cust_df['Material name'] == selected_prod].copy()
                m_plot = p_plot.groupby(p_plot['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
                m_plot['ds'] = m_plot['ds'].dt.to_timestamp()
                m_plot = m_plot.rename(columns={'Order qty.(A)': 'y'})

                m = Prophet(yearly_seasonality=True).fit(m_plot)
                fcst = m.predict(m.make_future_dataframe(periods=12, freq='MS'))

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=m_plot['ds'], y=m_plot['y'], name="Actual Demand", line=dict(color='#1f77b4', width=3)))
                fcst_26 = fcst[fcst['ds'].dt.year == 2026]
                fig.add_trace(go.Scatter(x=fcst_26['ds'], y=fcst_26['yhat'], name="AI Forecast", line=dict(dash='dash', color='#ff7f0e')))
                fig.update_layout(title=f"Supply Chain Timeline: {selected_prod}", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("📋 2026 Strategic Plan (Actuals + AI Projections)")
                months_26 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
                cols_26 = [m.strftime('%m/%Y') for m in months_26]
                pivot_list = []

                # Find the last month of actual data in 2026
                act_26_only = m_plot[m_plot['ds'].dt.year == 2026]
                max_act_date = act_26_only['ds'].max() if not act_26_only.empty else pd.Timestamp('2026-01-01')

                for p in top_prods:
                    avg_trend, y_val, rr_val = get_metrics(cust_df, p)
                    # Calculation Factor: YoY if available, otherwise Average Historical Trend
                    growth_factor = (y_val if y_val != 0 else (avg_trend/100)) + (adj_growth / 100)
                    
                    cies = cust_df[cust_df['Material name'] == p][cie_col].unique()
                    
                    for c in cies:
                        row = {'Product': p, 'CIE': c, 'YoY/Trend': f"{growth_factor*100:.1f}%"}
                        for m_date in months_26:
                            m_idx, m_str = m_date.month, m_date.strftime('%m/%Y')
                            
                            # Check actuals in Excel first
                            actual_val = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & 
                                                (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2026)]['Order qty.(A)'].sum()
                            
                            if actual_val > 0:
                                row[m_str] = actual_val
                            elif m_date > max_act_date:
                                # FORECAST LOGIC for Missing 2025 Data:
                                h25 = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & 
                                             (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2025)]['Order qty.(A)'].sum()
                                
                                if h25 > 0:
                                    # Scenario A: Use 2025 baseline + growth
                                    row[m_str] = round(h25 * (1 + growth_factor), 0)
                                else:
                                    # Scenario B (New Projects): Use 2026 Run-rate + Trend
                                    row[m_str] = round(rr_val * (1 + growth_factor), 0)
                            else:
                                row[m_str] = 0
                        pivot_list.append(row)
                
                if pivot_list:
                    res_df = pd.DataFrame(pivot_list)
                    # Grand Total Calculation
                    total_row = {'Product': 'GRAND TOTAL', 'CIE': '---', 'YoY/Trend': '---'}
                    for col in cols_26: total_row[col] = res_df[col].sum()
                    res_df = pd.concat([res_df, pd.DataFrame([total_row])], ignore_index=True)
                    
                    st.dataframe(res_df.style.apply(lambda x: ['background: #f0f2f6; font-weight: bold' if x['Product']=='GRAND TOTAL' else '' for _ in x], axis=1).format("{:,.0f}", subset=cols_26), use_container_width=True)
                    st.download_button("📥 Download Plan (CSV)", res_df.to_csv(index=False).encode('utf-8-sig'), "Strategic_Plan_2026.csv")
