import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. System Setup
st.set_page_config(page_title="AI Supply Chain Advisor 2026", layout="wide")

def process_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = [str(col).strip() for col in df.columns]
        date_col, qty_col = 'Requested deliv. date', 'Order qty.(A)'
        
        if date_col in df.columns and qty_col in df.columns:
            # Force data types to prevent missing rows
            df['ds'] = pd.to_datetime(df[date_col], errors='coerce')
            df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
            df = df.dropna(subset=['ds'])
            return df
        else:
            st.error(f"Error: Missing columns '{date_col}' or '{qty_col}'")
            return None
    except Exception as e:
        st.error(f"Read Error: {e}")
        return None

def get_metrics(cust_df, prod_name):
    p_df = cust_df[cust_df['Material name'] == prod_name].copy()
    if p_df.empty: return 0.0, 0.0, 0.0
    
    # Monthly aggregation of ALL history found in Excel
    monthly = p_df.groupby(p_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
    monthly['ds_ts'] = monthly['ds'].dt.to_timestamp()
    
    # Trend Calculation
    monthly['Pct_Change'] = monthly['Order qty.(A)'].pct_change() * 100
    avg_trend = monthly['Pct_Change'].mean() if not monthly['Pct_Change'].empty else 0.0
    
    # 2026 Run-rate (Average of months that have passed in 2026)
    df_26 = monthly[monthly['ds_ts'].dt.year == 2026]
    rr_26 = df_26['Order qty.(A)'].mean() if not df_26.empty else 0.0
    
    # YoY (2026 Actual vs 2025 same period)
    m_26 = df_26['ds_ts'].dt.month.tolist()
    act_25 = monthly[(monthly['ds_ts'].dt.year == 2025) & (monthly['ds_ts'].dt.month.isin(m_26))]['Order qty.(A)'].sum()
    yoy = (df_26['Order qty.(A)'].sum() - act_25) / act_25 if act_25 > 0 else 0.0
    
    return avg_trend, yoy, rr_26

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
            # Pareto 85% based on M USD
            rev = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
            top_prods = rev[rev['M USD'].cumsum() / rev['M USD'].sum() <= 0.86]['Material name'].unique()[:20]

            tab1, tab2 = st.tabs(["📊 Performance Audit", "📋 2026 Strategic Plan"])

            with tab1:
                selected_prod = st.selectbox("Product Audit:", top_prods)
                trnd, yoy, rr = get_metrics(cust_df, selected_prod)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Historical Trend", f"{trnd:.1f}%")
                m2.metric("YoY Growth", f"{yoy*100:.1f}%")
                m3.metric("2026 Run-rate", f"{rr:,.0f}")

                # AI Forecasting
                p_plot = cust_df[cust_df['Material name'] == selected_prod].copy()
                m_plot = p_plot.groupby(p_plot['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
                m_plot['ds'] = m_plot['ds'].dt.to_timestamp()
                m_plot = m_plot.rename(columns={'Order qty.(A)': 'y'})
                
                model = Prophet(yearly_seasonality=True).fit(m_plot)
                fcst = model.predict(model.make_future_dataframe(periods=12, freq='MS'))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=m_plot['ds'], y=m_plot['y'], name="Actual (from Excel)", line=dict(color='blue', width=3)))
                fcst_26 = fcst[fcst['ds'].dt.year == 2026]
                fig.add_trace(go.Scatter(x=fcst_26['ds'], y=fcst_26['yhat'], name="AI Forecast", line=dict(dash='dash', color='orange')))
                st.plotly_chart(fig, use_container_width=True)

                # --- THE MISSING VARIANCE TABLE & COMMENTS ---
                st.subheader("🔢 2026 Variance Analysis (Actual vs AI)")
                act_26 = m_plot[m_plot['ds'].dt.year == 2026]
                v_df = pd.merge(act_26, fcst_26[['ds', 'yhat']], on='ds', how='inner')
                
                if not v_df.empty:
                    v_df['Var %'] = ((v_df['y'] - v_df['yhat']) / v_df['yhat']) * 100
                    st.dataframe(v_df.style.format({'ds': lambda x: x.strftime('%m/%Y'), 'y': '{:,.0f}', 'yhat': '{:,.0f}', 'Var %': '{:+.1f}%'}), use_container_width=True)
                    
                    last_v = v_df['Var %'].iloc[-1]
                    st.markdown("### 💡 AI Strategic Commentary")
                    if last_v > 15:
                        st.warning(f"Demand is **{last_v:.1f}% above AI Forecast**. Urgently check component lead times.")
                    elif last_v < -15:
                        st.error(f"Demand is **{abs(last_v):.1f}% below AI Forecast**. Consider reducing stock orders.")
                    else:
                        st.success("Demand is stable and matching AI projections. No immediate action required.")

            with tab2:
                st.subheader("📋 2026 Full Year Strategic Plan")
                months_26 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
                cols_26 = [m.strftime('%m/%Y') for m in months_26]
                pivot_list = []
                last_act = m_plot['ds'].max() if not m_plot.empty else pd.Timestamp('2026-01-01')

                for p in top_prods:
                    p_trnd, p_yoy, p_rr = get_metrics(cust_df, p)
                    factor = (p_yoy if p_yoy != 0 else (p_trnd/100)) + (adj_growth / 100)
                    cies = cust_df[cust_df['Material name'] == p][cie_col].unique()
                    
                    for c in cies:
                        row = {'Product': p, 'CIE': c, 'Growth': f"{factor*100:.1f}%"}
                        for m_date in months_26:
                            m_idx, m_str = m_date.month, m_date.strftime('%m/%Y')
                            # Check actuals in Excel
                            act = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2026)]['Order qty.(A)'].sum()
                            
                            if act > 0: row[m_str] = act
                            elif m_date > last_act:
                                h25 = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2025)]['Order qty.(A)'].sum()
                                # Fallback: If 2025 is 0, use 2026 Average (Run-rate)
                                row[m_str] = round(h25 * (1 + factor), 0) if h25 > 0 else round(p_rr * (1 + factor), 0)
                            else: row[m_str] = 0
                        pivot_list.append(row)
                
                if pivot_list:
                    res_df = pd.DataFrame(pivot_list)
                    total_row = {'Product': 'GRAND TOTAL', 'CIE': '---', 'Growth': '---'}
                    for col in cols_26: total_row[col] = res_df[col].sum()
                    res_df = pd.concat([res_df, pd.DataFrame([total_row])], ignore_index=True)
                    st.dataframe(res_df.style.apply(lambda x: ['background: #e6f3ff; font-weight: bold' if x['Product']=='GRAND TOTAL' else '' for _ in x], axis=1).format("{:,.0f}", subset=cols_26), use_container_width=True)
