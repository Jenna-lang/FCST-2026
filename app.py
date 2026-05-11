import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. System Config
st.set_page_config(page_title="AI Supply Chain Advisor 2026", layout="wide")

# --- CORE LOGIC FUNCTIONS ---
def get_actual_avg_qty(df, year, quarter, prod, cie_col, cie_val):
    temp = df[(df['Material name'] == prod) & 
              (df[cie_col] == cie_val) & 
              (df['ds'].dt.year == year) & 
              (df['ds'].dt.quarter == quarter)].copy()
    monthly_sum = temp.groupby(temp['ds'].dt.month)['Order qty.(A)'].sum()
    actual_months = monthly_sum[monthly_sum > 0]
    return actual_months.mean() if not actual_months.empty else 0.0

def get_quarterly_growth_logic(cust_df, prod, cie_col, cie_val):
    df_26 = cust_df[cust_df['ds'].dt.year == 2026].copy()
    if df_26.empty: return 0.0
    valid_26 = df_26[df_26['Order qty.(A)'] > 0]
    if valid_26.empty: return 0.0
    latest_q_26 = valid_26['ds'].dt.quarter.max()
    avg_26 = get_actual_avg_qty(cust_df, 2026, latest_q_26, prod, cie_col, cie_val)
    avg_25 = get_actual_avg_qty(cust_df, 2025, latest_q_26, prod, cie_col, cie_val)
    return min(max((avg_26 / avg_25) - 1, -0.5), 0.5) if avg_25 > 0 else 0.0

def process_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = [str(col).strip() for col in df.columns]
        if 'Requested deliv. date' in df.columns:
            df['ds'] = pd.to_datetime(df['Requested deliv. date'], errors='coerce')
            df['Order qty.(A)'] = pd.to_numeric(df['Order qty.(A)'], errors='coerce').fillna(0)
            return df.dropna(subset=['ds'])
        return None
    except: return None

# --- MAIN UI ---
st.sidebar.header("📁 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload AICheck.xlsx", type=['xlsx'])

if uploaded_file:
    df = process_data(uploaded_file)
    if df is not None:
        all_cols = df.columns.tolist()
        cust_col = st.sidebar.selectbox("Customer Column:", all_cols, index=0)
        cie_col = st.sidebar.selectbox("CIE / Color Code Column:", all_cols, index=1)
        adj_growth = st.sidebar.slider("Manual Growth Adjustment (%)", -50, 50, 0)
        selected_cust = st.sidebar.selectbox("Select Target Customer:", ["-- Select --"] + sorted(df[cust_col].unique().tolist()))

        if selected_cust != "-- Select --":
            cust_df = df[df[cust_col] == selected_cust].copy()
            rev = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
            rev['Cum%'] = rev['M USD'].cumsum() / rev['M USD'].sum()
            top_prods = rev[rev['Cum%'] <= 0.86]['Material name'].unique()

            # --- TẠO TAB TẠI ĐÂY ---
            tab1, tab2, tab3 = st.tabs(["📊 Performance Audit", "📋 2026 Strategic Plan", "🧪 Model Testing"])
            auto_adjustments = {}

            with tab1:
                st.subheader("🎯 Pareto 85% & Variance Audit")
                selected_prod = st.selectbox("Product Audit:", top_prods)
                
                p_plot = cust_df[cust_df['Material name'] == selected_prod].groupby(cust_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
                p_plot['ds'] = p_plot['ds'].dt.to_timestamp()
                p_plot = p_plot.rename(columns={'Order qty.(A)': 'y'})
                
                if len(p_plot) > 2:
                    model = Prophet(yearly_seasonality=True).fit(p_plot)
                    fcst = model.predict(model.make_future_dataframe(periods=12, freq='MS'))
                    fcst_26 = fcst[fcst['ds'].dt.year == 2026].copy()
                    v_df = pd.merge(p_plot[p_plot['ds'].dt.year == 2026], fcst_26[['ds', 'yhat']], on='ds')
                    
                    avg_v = ((v_df['y'] - v_df['yhat']) / v_df['yhat']).mean() if not v_df.empty else 0
                    auto_adjustments[selected_prod] = avg_v
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=p_plot['ds'], y=p_plot['y'], name="Actual"))
                    fig.add_trace(go.Scatter(x=fcst_26['ds'], y=fcst_26['yhat']*(1+avg_v), name="Adjusted FCST", line=dict(dash='dash', color='orange')))
                    st.plotly_chart(fig, use_container_width=True)

                    # KHÔI PHỤC BẢNG VARIANCE CHI TIẾT
                    st.subheader("🔢 Variance Details")
                    v_df['Variance %'] = ((v_df['y'] - v_df['yhat']) / v_df['yhat']) * 100
                    st.dataframe(v_df.rename(columns={'ds':'Month', 'y':'Actual', 'yhat':'AI FCST'}).style.format({'Variance %': '{:+.1f}%'}))

            with tab2:
                st.subheader("📋 2026 Strategic Plan (Tiered Adjustment)")
                months_26 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
                last_act_date = df[df['Order qty.(A)'] > 0]['ds'].max()
                pivot_list = []

                for p in top_prods:
                    p_var = auto_adjustments.get(p, 0.0)
                    for c in cust_df[cust_df['Material name']==p][cie_col].unique():
                        row = {'Product': p, 'CIE': str(c)}
                        for m_date in months_26:
                            gap = (m_date.year - last_act_date.year)*12 + (m_date.month - last_act_date.month)
                            thresh = 0.2 if gap==2 else (0.3 if gap==3 else 0.5)
                            offset = p_var if abs(p_var) > thresh else 0
                            
                            act = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & (cust_df['ds']==m_date)]['Order qty.(A)'].sum()
                            if act > 0: row[m_date.strftime('%m/%Y')] = act
                            elif m_date > last_act_date:
                                avg25 = get_actual_avg_qty(cust_df, 2025, (m_date.month-1)//3+1, p, cie_col, c)
                                row[m_date.strftime('%m/%Y')] = round(avg25 * (1 + offset + (adj_growth/100)), 0)
                        pivot_list.append(row)
                st.dataframe(pd.DataFrame(pivot_list), use_container_width=True)

            with tab3:
                st.header("🧪 Compliance Audit (M+2/3/4)")
                test_results = []
                for p in top_prods:
                    v = auto_adjustments.get(p, 0.0) * 100
                    test_results.append({
                        "Product": p, "Variance": v,
                        "M+2 (20%)": "✅ Pass" if abs(v)<=20 else "❌ Fail",
                        "M+3 (30%)": "✅ Pass" if abs(v)<=30 else "❌ Fail",
                        "M+4 (50%)": "✅ Pass" if abs(v)<=50 else "❌ Fail"
                    })
                st.table(pd.DataFrame(test_results))
