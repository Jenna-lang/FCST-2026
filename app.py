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
            df['ds'] = pd.to_datetime(df[date_col], errors='coerce')
            df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
            df = df.dropna(subset=['ds'])
            return df
        return None
    except Exception as e:
        st.error(f"File Read Error: {e}")
        return None

def get_precise_metrics(cust_df, prod_name, cie_col_name, cie_val=None):
    if cie_val:
        p_df = cust_df[(cust_df['Material name'] == prod_name) & (cust_df[cie_col_name] == cie_val)].copy()
    else:
        p_df = cust_df[cust_df['Material name'] == prod_name].copy()
        
    if p_df.empty: return 0.0, 0.0, 0.0
    
    monthly = p_df.groupby(p_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
    monthly['ds_ts'] = monthly['ds'].dt.to_timestamp()
    
    # Metrics calculation
    monthly['Pct_Change'] = monthly['Order qty.(A)'].pct_change() * 100
    avg_trend = monthly['Pct_Change'].mean() if not monthly['Pct_Change'].empty else 0.0
    
    df_26 = monthly[monthly['ds_ts'].dt.year == 2026]
    ma_26 = df_26['Order qty.(A)'].tail(3).mean() if not df_26.empty else 0.0
    
    m_26_list = df_26['ds_ts'].dt.month.tolist()
    act_25 = monthly[(monthly['ds_ts'].dt.year == 2025) & (monthly['ds_ts'].dt.month.isin(m_26_list))]['Order qty.(A)'].sum()
    yoy_raw = (df_26['Order qty.(A)'].sum() - act_25) / act_25 if act_25 > 0 else 0.0
    yoy_final = min(yoy_raw, 0.5) # Growth Cap
    
    return avg_trend, yoy_final, ma_26

# --- MAIN UI ---
st.sidebar.header("📁 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload AICheck.xlsx", type=['xlsx'])

if uploaded_file:
    df = process_data(uploaded_file)
    if df is not None:
        c_cols = [c for c in df.columns if 'Customer' in c]
        cie_col_name = df.columns[1] # Detecting CIE column
        cust_col = c_cols[0] if c_cols else df.columns[0]
        
        adj_growth = st.sidebar.slider("Manual Adjustment (%)", -50, 50, 0)
        selected_cust = st.sidebar.selectbox("Select Customer:", ["-- Select --"] + sorted(df[cust_col].unique()))

        if selected_cust != "-- Select --":
            cust_df = df[df[cust_col] == selected_cust].copy()
            
            # --- PARETO 85% ---
            rev = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
            rev['CumSum'] = rev['M USD'].cumsum() / rev['M USD'].sum()
            top_prods = rev[rev['CumSum'] <= 0.86]['Material name'].unique()[:20]

            tab1, tab2 = st.tabs(["📊 Performance Audit", "📋 2026 Strategic Plan"])

            with tab1:
                selected_prod = st.selectbox("Product Audit:", top_prods)
                trnd, yoy, ma = get_precise_metrics(cust_df, selected_prod, cie_col_name)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Historical Trend", f"{trnd:.1f}%")
                m2.metric("YoY Growth", f"{yoy*100:.1f}%")
                m3.metric("Moving Avg (3M)", f"{ma:,.0f}")

                # AI Forecasting Chart
                p_plot = cust_df[cust_df['Material name'] == selected_prod].copy()
                m_plot = p_plot.groupby(p_plot['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
                m_plot['ds'] = m_plot['ds'].dt.to_timestamp()
                m_plot = m_plot.rename(columns={'Order qty.(A)': 'y'})
                
                model = Prophet(yearly_seasonality=True).fit(m_plot)
                fcst = model.predict(model.make_future_dataframe(periods=12, freq='MS'))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=m_plot['ds'], y=m_plot['y'], name="Actual", line=dict(color='blue')))
                fcst_26 = fcst[fcst['ds'].dt.year == 2026]
                fig.add_trace(go.Scatter(x=fcst_26['ds'], y=fcst_26['yhat'], name="AI Forecast", line=dict(dash='dash', color='orange')))
                st.plotly_chart(fig, use_container_width=True)

                # --- VARIANCE & COMMENTARY ---
                st.subheader("🔢 2026 Variance Analysis")
                act_26 = m_plot[m_plot['ds'].dt.year == 2026]
                v_df = pd.merge(act_26, fcst_26[['ds', 'yhat']], on='ds', how='inner')
                if not v_df.empty:
                    v_df['Var %'] = ((v_df['y'] - v_df['yhat']) / v_df['yhat']) * 100
                    st.dataframe(v_df.style.format({'ds': lambda x: x.strftime('%m/%Y'), 'y': '{:,.0f}', 'yhat': '{:,.0f}', 'Var %': '{:+.1f}%'}), use_container_width=True)
                    
                    last_v = v_df['Var %'].iloc[-1]
                    st.markdown("### 💡 AI Strategic Commentary")
                    if last_v > 15:
                        st.warning(f"Demand is {last_v:.1f}% higher than forecast. Action: Check IC supply chain.")
                    elif last_v < -15:
                        st.error(f"Demand is {abs(last_v):.1f}% lower than forecast. Action: Manage inventory levels.")
                    else:
                        st.success("Demand is stable and matching AI projections.")

            with tab2:
                st
