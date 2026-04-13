import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. System Config
st.set_page_config(page_title="AI Supply Chain Advisor 2026", layout="wide")

# --- CORE LOGIC FUNCTIONS (STRICTLY UNCHANGED) ---

def get_actual_avg_qty(df, year, quarter, prod, cie_col, cie_val):
    """Bảo toàn logic cũ: Tính trung bình các tháng > 0 trong quý."""
    temp = df[(df['Material name'] == prod) & 
              (df[cie_col] == cie_val) & 
              (df['ds'].dt.year == year) & 
              (df['ds'].dt.quarter == quarter)].copy()
    
    monthly_sum = temp.groupby(temp['ds'].dt.month)['Order qty.(A)'].sum()
    actual_months = monthly_sum[monthly_sum > 0]
    
    if actual_months.empty:
        return 0.0
    return actual_months.mean()

def get_quarterly_growth_logic(cust_df, prod, cie_col, cie_val):
    """Bảo toàn logic cũ: Tăng trưởng dựa trên Q gần nhất 2026 vs 2025."""
    df_26 = cust_df[cust_df['ds'].dt.year == 2026].copy()
    if df_26.empty: return 0.0
    
    valid_26 = df_26[df_26['Order qty.(A)'] > 0]
    if valid_26.empty: return 0.0
    
    latest_q_26 = valid_26['ds'].dt.quarter.max()
    
    avg_26 = get_actual_avg_qty(cust_df, 2026, latest_q_26, prod, cie_col, cie_val)
    avg_25 = get_actual_avg_qty(cust_df, 2025, latest_q_26, prod, cie_col, cie_val)
    
    if avg_25 > 0:
        growth = (avg_26 / avg_25) - 1
        return min(max(growth, -0.5), 0.5)
    return 0.0

def process_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = [str(col).strip() for col in df.columns]
        if 'Requested deliv. date' in df.columns and 'Order qty.(A)' in df.columns:
            df['ds'] = pd.to_datetime(df['Requested deliv. date'], errors='coerce')
            df['Order qty.(A)'] = pd.to_numeric(df['Order qty.(A)'], errors='coerce').fillna(0)
            return df.dropna(subset=['ds'])
        return None
    except Exception as e:
        st.error(f"File Error: {e}")
        return None

# --- MAIN UI ---
st.sidebar.header("📁 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload AICheck.xlsx", type=['xlsx'])

if uploaded_file:
    df = process_data(uploaded_file)
    if df is not None:
        all_cols = df.columns.tolist()
        
        cust_suggest = next((c for c in all_cols if 'customer' in c.lower()), all_cols[0])
        cust_col = st.sidebar.selectbox("Customer Column:", all_cols, index=all_cols.index(cust_suggest))
        
        cie_suggest = all_cols[1] if len(all_cols) > 1 else all_cols[0]
        cie_col = st.sidebar.selectbox("CIE / Color Code Column:", all_cols, index=all_cols.index(cie_suggest))
        
        adj_growth = st.sidebar.slider("Growth Adjustment (%)", -50, 50, 0)
        
        selected_cust = st.sidebar.selectbox("Select Target Customer:", ["-- Select --"] + sorted(df[cust_col].unique().tolist()))

        if selected_cust != "-- Select --":
            cust_df = df[df[cust_col] == selected_cust].copy()
            
            # Pareto 85% Calculation
            rev = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
            rev['Cum%'] = rev['M USD'].cumsum() / rev['M USD'].sum()
            pareto_df = rev[rev['Cum%'] <= 0.86].copy()
            top_prods = pareto_df['Material name'].unique()

            tab1, tab2 = st.tabs(["📊 Performance Audit", "📋 2026 Strategic Plan"])

            with tab1:
                # 1. Pareto Label (Expander)
                with st.expander("🎯 85% Revenue Contribution (Pareto List)"):
                    st.write(f"The following {len(top_prods)} products account for 85% of revenue:")
                    st.dataframe(pareto_df[['Material name', 'M USD', 'Cum%']].style.format({'M USD': '${:,.2f}', 'Cum%': '{:.1%}'}), use_container_width=True)

                selected_prod = st.selectbox("Product Audit:", top_prods)
                sample_cies = cust_df[cust_df['Material name'] == selected_prod][cie_col].unique()
                s_cie = sample_cies[0]
                
                q_growth = get_quarterly_growth_logic(cust_df, selected_prod, cie_col, s_cie)
                f_growth = q_growth + (adj_growth/100)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Quarterly Growth", f"{q_growth*100:.1f}%")
                m2.metric("Adjusted Growth", f"{f_growth*100:.1f}%")
                m3.metric("CIE Items", len(sample_cies))

                p_plot = cust_df[cust_df['Material name'] == selected_prod].groupby(cust_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
                p_plot['ds'] = p_plot['ds'].dt.to_timestamp()
                p_plot = p_plot.rename(columns={'Order qty.(A)': 'y'})
                
                if len(p_plot) > 2:
                    model = Prophet(yearly_seasonality=True).fit(p_plot)
                    fcst = model.predict(model.make_future_dataframe(periods=12, freq='MS'))
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=p_plot['ds'], y=p_plot['y'], name="Actual", line=dict(color='blue')))
                    fcst_26 = fcst[fcst['ds'].dt.year == 2026]
                    fig.add_trace(go.Scatter(x=fcst_26['ds'], y=fcst_26['yhat'], name="AI Forecast", line=dict(dash='dash', color='orange')))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 2. Updated Table Headers & Average Label
                    st.subheader("🔢 Actual vs AI Variance")
                    act_26 = p_plot[p_plot['ds'].dt.year == 2026]
                    v_df = pd.merge(act_26, fcst_26[['ds', 'yhat']], on='ds', how='inner')
                    
                    if not v_df.empty:
                        v_df['Variance %'] = ((v_df['y'] - v_df['yhat']) / v_df['yhat']) * 100
                        
                        # Rename columns as requested
                        v_df = v_df.rename(columns={
                            'ds': 'Month Code',
                            'y': 'Actual Order Quantity',
                            'yhat': 'AI FCST Quantity'
                        })
                        
                        avg_row = pd.DataFrame({
                            'Month Code': ["AVERAGE"], 
                            'Actual Order Quantity': [v_df['Actual Order Quantity'].mean()], 
                            'AI FCST Quantity': [v_df['AI FCST Quantity'].mean()], 
                            'Variance %': [v_df['Variance %'].mean()]
                        })
                        v_display = pd.concat([v_df, avg_row], ignore_index=True)
                        
                        st.dataframe(v_display.style.format({
                            'Month Code': lambda x: x.strftime('%m/%Y') if hasattr(x, 'strftime') else x,
                            'Actual Order Quantity': '{:,.0f}', 
                            'AI FCST Quantity': '{:,.0f}', 
                            'Variance %': '{:+.1f}%'
                        }).apply(lambda x: ['background: #f0f2f6; font-weight: bold'] * len(x) if x['Month Code'] == "AVERAGE" else [''] * len(x), axis=1), use_container_width=True)

            with tab2:
                # Tab 2 remains with original logic
                st.subheader(f"📋 2026 Strategic Plan for {selected_cust}")
                months_26 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
                cols_26 = [m.strftime('%m/%Y') for m in months_26]
                pivot_list = []
                last_act_date = df[df['Order qty.(A)'] > 0]['ds'].max()

                for p in top_prods:
                    product_cies = cust_df[cust_df['Material name'] == p][cie_col].unique()
                    for c in product_cies:
                        q_grow = get_quarterly_growth_logic(cust_df, p, cie_col, c)
                        f_grow = q_grow + (adj_growth/100)
                        
                        row = {'Product': p, 'CIE': str(c), 'Growth Rate': f"{f_grow*100:.1f}%"}
                        for m_date in months_26:
                            m_idx, m_str = m_date.month, m_date.strftime('%m/%Y')
                            q_idx = (m_idx - 1) // 3 + 1
                            act_val = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2026)]['Order qty.(A)'].sum()
                            
                            if act_val > 0:
                                row[m_str] = act_val
                            elif m_date > last_act_date:
                                avg_25 = get_actual_avg_qty(cust_df, 2025, q_idx, p, cie_col, c)
                                row[m_str] = round(avg_25 * (1 + f_grow), 0)
                            else:
                                row[m_str] = 0
                        pivot_list.append(row)
                
                if pivot_list:
                    res_df = pd.DataFrame(pivot_list)
                    total_row = {'Product': 'GRAND TOTAL', 'CIE': '', 'Growth Rate': ''}
                    for col in cols_26: total_row[col] = res_df[col].sum()
                    res_df = pd.concat([res_df, pd.DataFrame([total_row])], ignore_index=True)
                    st.dataframe(res_df.style.apply(lambda x: ['background: #e6f3ff; font-weight: bold'] * len(x) if x['Product']=='GRAND TOTAL' else [''] * len(x), axis=1).format("{:,.0f}", subset=cols_26), use_container_width=True)
                    st.download_button("📥 Download Plan (CSV)", data=res_df.to_csv(index=False).encode('utf-8-sig'), file_name="Strategic_Plan_2026.csv")
