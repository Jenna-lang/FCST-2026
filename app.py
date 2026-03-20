import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. System Configuration
st.set_page_config(page_title="AI Supply Chain Advisor 2026", layout="wide")

# --- DATA PROCESSING FUNCTIONS ---
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
            st.error(f"Missing columns: '{date_col}' or '{qty_col}' in Excel file!")
            return None
    except Exception as e:
        st.error(f"File Error: {e}")
        return None

def calculate_full_metrics(cust_df, prod_name):
    p_df = cust_df[cust_df['Material name'] == prod_name].copy()
    if p_df.empty: return 0.0, 0.0, 0.0
    
    monthly_all = p_df.groupby(p_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
    monthly_all['ds_ts'] = monthly_all['ds'].dt.to_timestamp()
    
    # 1. Avg Growth (Historical)
    monthly_all['Pct_Change'] = monthly_all['Order qty.(A)'].pct_change() * 100
    avg_growth = monthly_all['Pct_Change'].mean()
    
    # 2. 2026 Performance
    df_2026 = monthly_all[monthly_all['ds_ts'].dt.year == 2026]
    act_2026_sum = df_2026['Order qty.(A)'].sum()
    run_rate_26 = df_2026['Order qty.(A)'].mean() if not df_2026.empty else 0.0
    
    # 3. YoY Growth (2026 vs 2025 same period)
    months_active_26 = df_2026['ds_ts'].dt.month.tolist()
    act_2025_same = monthly_all[(monthly_all['ds_ts'].dt.year == 2025) & 
                                (monthly_all['ds_ts'].dt.month.isin(months_active_26))]['Order qty.(A)'].sum()
    yoy_growth = (act_2026_sum - act_2025_same) / act_2025_same if act_2025_same > 0 else 0.0
    
    return avg_growth, yoy_growth, run_rate_26

# --- SIDEBAR ---
st.sidebar.header("📁 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload AICheck.xlsx", type=['xlsx'])

if uploaded_file:
    df = process_data(uploaded_file)
    if df is not None:
        cust_col = st.sidebar.selectbox("Customer Column:", [c for c in df.columns if 'Customer' in c] or [df.columns[0]])
        cie_col = st.sidebar.selectbox("CIE/Color Column:", [c for c in df.columns if 'CIE' in c] or [df.columns[1]])
        adj_growth = st.sidebar.slider("Manual Adjustment (%)", -50, 50, 0)
        selected_cust = st.sidebar.selectbox("Select Customer:", ["-- Select --"] + sorted(df[cust_col].unique()))

        if selected_cust != "-- Select --":
            cust_df = df[df[cust_col] == selected_cust].copy()
            
            # Pareto 85% Logic
            rev = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
            rev['Cum_Pct'] = rev['M USD'].cumsum() / rev['M USD'].sum()
            top_prods = rev[rev['Cum_Pct'] <= 0.86]['Material name'].unique()[:20]

            tab1, tab2 = st.tabs(["📊 Performance & AI Audit", "📋 2026 Strategic Plan"])

            with tab1:
                selected_prod = st.selectbox("Product Audit:", top_prods)
                avg_g, yoy_g, r_rate = calculate_full_metrics(cust_df, selected_prod)
                
                # Metrics Display
                m1, m2, m3 = st.columns(3)
                m1.metric("Avg Monthly Growth", f"{avg_g:.1f}%")
                m2.metric("YoY Growth (26 vs 25)", f"{yoy_g*100:.1f}%")
                m3.metric("Run-rate 2026", f"{r_rate:,.0f}")

                # Charting (2023 -> 2026)
                p_df_plot = cust_df[cust_df['Material name'] == selected_prod].copy()
                df_monthly = p_df_plot.groupby(p_df_plot['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
                df_monthly['ds'] = df_monthly['ds'].dt.to_timestamp()
                df_monthly = df_monthly.rename(columns={'Order qty.(A)': 'y'})

                m = Prophet(yearly_seasonality=True).fit(df_monthly)
                future = m.make_future_dataframe(periods=12, freq='MS')
                fcst = m.predict(future)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_monthly['ds'], y=df_monthly['y'], name="Actual Demand", line=dict(color='#1f77b4', width=3)))
                fcst_26 = fcst[fcst['ds'].dt.year == 2026]
                fig.add_trace(go.Scatter(x=fcst_26['ds'], y=fcst_26['yhat'], name="AI Forecast", line=dict(dash='dash', color='#ff7f0e')))
                fig.update_layout(title=f"Demand History & AI Forecast: {selected_prod}", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

                # --- ACTUAL VS AI VARIANCE ---
                st.subheader("🔢 2026 Variance Analysis (Actual vs AI)")
                act_2026 = df_monthly[df_monthly['ds'].dt.year == 2026].copy()
                comparison = pd.merge(act_2026, fcst_26[['ds', 'yhat']], on='ds', how='left')
                if not comparison.empty:
                    comparison['Var %'] = ((comparison['y'] - comparison['yhat']) / comparison['yhat']) * 100
                    st.dataframe(comparison.style.format({'ds': lambda x: x.strftime('%m/%Y'), 'y': '{:,.0f}', 'yhat': '{:,.0f}', 'Var %': '{:+.1f}%'}), use_container_width=True)
                    
                    # AI ADVICE
                    diff = comparison['Var %'].iloc[-1]
                    if diff > 15:
                        st.warning(f"💡 **AI Advice**: Demand is **{diff:.1f}% higher** than forecast. Check IC supplier capacity immediately.")
                    elif diff < -15:
                        st.error(f"💡 **AI Advice**: Demand is **{abs(diff):.1f}% lower** than forecast. Review inventory to avoid overstock.")
                    else:
                        st.success("💡 **AI Advice**: Actual demand is closely aligned with AI forecast. Maintain current plan.")

            with tab2:
                st.subheader("📋 2026 Strategic Plan (Full View)")
                months_26 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
                cols_26 = [m.strftime('%m/%Y') for m in months_26]
                pivot_list = []

                for p in top_prods:
                    _, y_val, rr_val = calculate_full_metrics(cust_df, p)
                    final_f = 1 + y_val + (adj_growth / 100)
                    cies = cust_df[cust_df['Material name'] == p][cie_col].unique()
                    
                    for c in cies:
                        row = {'Product': p, 'CIE/Color': c, 'YoY Growth': f"{y_val*100:.1f}%"}
                        for m_date in months_26:
                            m_idx, m_str = m_date.month, m_date.strftime('%m/%Y')
                            act_v = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & 
                                            (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2026)]['Order qty.(A)'].sum()
                            
                            if act_v > 0:
                                row[m_str] = act_v
                            elif m_date > df_monthly['ds'].max():
                                h25 = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & 
                                              (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2025)]['Order qty.(A)'].sum()
                                # Logic: Use 2025 * Growth. If New Project (no 2025 data), use 2026 Run-rate.
                                row[m_str] = round(h25 * final_f, 0) if h25 > 0 else round(rr_val * final_f,
