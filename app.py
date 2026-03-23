import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. System Setup
st.set_page_config(page_title="AI Supply Chain Advisor 2026", layout="wide")

# Define the metric function FIRST
def get_precise_metrics(cust_df, prod_name, cie_col_name, cie_val=None):
    if cie_val:
        p_df = cust_df[(cust_df['Material name'] == prod_name) & (cust_df[cie_col_name] == cie_val)].copy()
    else:
        p_df = cust_df[cust_df['Material name'] == prod_name].copy()
        
    if p_df.empty: 
        return 0.0, 0.0, 0.0
    
    monthly = p_df.groupby(p_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
    monthly['ds_ts'] = monthly['ds'].dt.to_timestamp()
    
    monthly['Pct_Change'] = monthly['Order qty.(A)'].pct_change() * 100
    avg_trend = monthly['Pct_Change'].mean() if not monthly['Pct_Change'].empty else 0.0
    
    df_26 = monthly[monthly['ds_ts'].dt.year == 2026]
    ma_26 = df_26['Order qty.(A)'].tail(3).mean() if not df_26.empty else 0.0
    
    m_26_list = df_26['ds_ts'].dt.month.tolist()
    act_25 = monthly[(monthly['ds_ts'].dt.year == 2025) & (monthly['ds_ts'].dt.month.isin(m_26_list))]['Order qty.(A)'].sum()
    yoy_raw = (df_26['Order qty.(A)'].sum() - act_25) / act_25 if act_25 > 0 else 0.0
    yoy_final = min(yoy_raw, 0.5) 
    
    return avg_trend, yoy_final, ma_26

def process_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Required columns mapping
        date_col = 'Requested deliv. date'
        qty_col = 'Order qty.(A)'
        
        if date_col in df.columns and qty_col in df.columns:
            df['ds'] = pd.to_datetime(df[date_col], errors='coerce')
            df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
            df = df.dropna(subset=['ds'])
            return df
        else:
            st.error(f"Missing required columns: '{date_col}' or '{qty_col}'")
            return None
    except Exception as e:
        st.error(f"File Read Error: {e}")
        return None

# --- MAIN UI ---
st.sidebar.header("📁 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload AICheck.xlsx", type=['xlsx'])

if uploaded_file:
    df = process_data(uploaded_file)
    if df is not None:
        # Improved Customer Column Detection
        c_cols = [c for c in df.columns if 'Customer' in c or 'Cust' in c]
        
        if not c_cols:
            st.sidebar.error("Could not find a 'Customer' column in Excel.")
        else:
            cust_col = c_cols[0]
            cie_col_name = df.columns[1] # Assuming 2nd column is CIE
            
            adj_growth = st.sidebar.slider("Manual Adjustment (%)", -50, 50, 0)
            
            customer_list = sorted(df[cust_col].dropna().unique())
            selected_cust = st.sidebar.selectbox("Select Customer:", ["-- Select --"] + customer_list)

            if selected_cust != "-- Select --":
                cust_df = df[df[cust_col] == selected_cust].copy()
                
                # Pareto 85% Logic
                rev = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
                rev['CumSum'] = rev['M USD'].cumsum() / rev['M USD'].sum()
                top_prods = rev[rev['CumSum'] <= 0.86]['Material name'].unique()[:20]

                tab1, tab2 = st.tabs(["📊 Performance Audit", "📋 2026 Strategic Plan"])

                with tab1:
                    if len(top_prods) > 0:
                        selected_prod = st.selectbox("Product Audit:", top_prods)
                        trnd, yoy, ma = get_precise_metrics(cust_df, selected_prod, cie_col_name)
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Historical Trend", f"{trnd:.1f}%")
                        m2.metric("YoY Growth", f"{yoy*100:.1f}%")
                        m3.metric("Moving Avg (3M)", f"{ma:,.0f}")

                        # Chart logic
                        p_plot = cust_df[cust_df['Material name'] == selected_prod].copy()
                        m_plot = p_plot.groupby(p_plot['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
                        m_plot['ds'] = m_plot['ds'].dt.to_timestamp()
                        m_plot = m_plot.rename(columns={'Order qty.(A)': 'y'})
                        
                        model = Prophet(yearly_seasonality=True).fit(m_plot)
                        fcst = model.predict(model.make_future_dataframe(periods=12, freq='MS'))
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=m_plot['ds'], y=m_plot['y'], name="Actual", line=dict(color='blue', width=2)))
                        fcst_26 = fcst[fcst['ds'].dt.year == 2026]
                        fig.add_trace(go.Scatter(x=fcst_26['ds'], y=fcst_26['yhat'], name="AI Forecast", line=dict(dash='dash', color='orange')))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Variance Analysis
                        st.subheader("🔢 2026 Variance Analysis")
                        act_26 = m_plot[m_plot['ds'].dt.year == 2026]
                        v_df = pd.merge(act_26, fcst_26[['ds', 'yhat']], on='ds', how='inner')
                        if not v_df.empty:
                            v_df['Var %'] = ((v_df['y'] - v_df['yhat']) / v_df['yhat']) * 100
                            st.dataframe(v_df.style.format({'ds': lambda x: x.strftime('%m/%Y'), 'y': '{:,.0f}', 'yhat': '{:,.0f}', 'Var %': '{:+.1f}%'}), use_container_width=True)
                    else:
                        st.warning("No top products found for Pareto 85% calculation.")

                with tab2:
                    st.subheader("📋 2026 Full Year Strategic Plan")
                    months_26 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
                    cols_26 = [m.strftime('%m/%Y') for m in months_26]
                    pivot_list = []
                    last_act_date = df[df['Order qty.(A)'] > 0]['ds'].max()

                    for p in top_prods:
                        cies = cust_df[cust_df['Material name'] == p][cie_col_name].unique()
                        for c in cies:
                            _, p_yoy, p_ma = get_precise_metrics(cust_df, p, cie_col_name, c)
                            final_f = p_yoy + (adj_growth / 100)
                            row = {'Product': p, 'CIE': c, 'Growth': f"{final_f*100:.1f}%"}
                            for m_date in months_26:
                                m_idx, m_str = m_date.month, m_date.strftime('%m/%Y')
                                act_val = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col_name]==c) & 
                                                  (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2026)]['Order qty.(A)'].sum()
                                if act_val > 0:
                                    row[m_str] = act_val
                                elif m_date > last_act_date:
                                    h25 = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col_name]==c) & 
                                                  (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2025)]['Order qty.(A)'].sum()
                                    base_val = h25 if h25 > 0 else p_ma
                                    row[m_str] = round(base_val * (1 + final_f), 0)
                                else:
                                    row[m_str] = 0
                            pivot_list.append(row)
                    
                    if pivot_list:
                        res_df = pd.DataFrame(pivot_list)
                        total_row = {'Product': 'GRAND TOTAL', 'CIE': '---', 'Growth': '---'}
                        for col in cols_26: total_row[col] = res_df[col].sum()
                        res_df = pd.concat([res_df, pd.DataFrame([total_row])], ignore_index=True)
                        st.dataframe(res_df.style.apply(lambda x: ['background: #e6f3ff; font-weight: bold' if x['Product']=='GRAND TOTAL' else '' for _ in x], axis=1).format("{:,.0f}", subset=cols_26), use_container_width=True)
                        st.download_button("📥 Download Plan (CSV)", data=res_df.to_csv(index=False).encode('utf-8-sig'), file_name="Strategic_Plan_2026.csv")
