import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. System Setup
st.set_page_config(page_title="AI Supply Chain Advisor 2026", layout="wide")

# --- CORE LOGIC ---

def get_actual_avg_qty(df, year, quarter, prod, cie_col, cie_val):
    """Tính trung bình thực tế các tháng có số > 0 trong một Quý."""
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
    """So sánh TB quý gần nhất 2026 vs cùng kỳ 2025."""
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
        # Nhận diện cột
        all_cols = df.columns.tolist()
        cust_col = next((c for c in all_cols if 'customer' in c.lower()), all_cols[0])
        cie_col = all_cols[1] if all_cols[1] != cust_col else all_cols[0]
        
        st.sidebar.info(f"Detected CIE Column: **{cie_col}**") # Hiển thị để Jenna kiểm tra
        
        adj_growth = st.sidebar.slider("Global Adjustment (%)", -50, 50, 0)
        selected_cust = st.sidebar.selectbox("Select Customer:", ["-- Select --"] + sorted(df[cust_col].unique().tolist()))

        if selected_cust != "-- Select --":
            cust_df = df[df[cust_col] == selected_cust].copy()
            
            # Pareto 85%
            rev = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
            top_prods = rev[rev['M USD'].cumsum() / rev['M USD'].sum() <= 0.86]['Material name'].unique()[:20]

            tab1, tab2 = st.tabs(["📊 Performance Audit", "📋 2026 Strategic Plan"])

            with tab1:
                selected_prod = st.selectbox("Select Product to Audit:", top_prods)
                
                # Audit dựa trên CIE đầu tiên của sản phẩm
                sample_cies = cust_df[cust_df['Material name'] == selected_prod][cie_col].unique()
                sample_cie = sample_cies[0]
                
                q_growth = get_quarterly_growth_logic(cust_df, selected_prod, cie_col, sample_cie)
                f_growth = q_growth + (adj_growth/100)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Quarterly Basis", f"{q_growth*100:.1f}%")
                m2.metric("Final Growth", f"{f_growth*100:.1f}%")
                
                # Prophet Forecast & Variance Analysis
                p_plot = cust_df[cust_df['Material name'] == selected_prod].copy()
                m_plot = p_plot.groupby(p_plot['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
                m_plot['ds'] = m_plot['ds'].dt.to_timestamp()
                m_plot = m_plot.rename(columns={'Order qty.(A)': 'y'})
                
                if len(m_plot) > 2:
                    model = Prophet(yearly_seasonality=True).fit(m_plot)
                    fcst = model.predict(model.make_future_dataframe(periods=12, freq='MS'))
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=m_plot['ds'], y=m_plot['y'], name="Actual", line=dict(color='blue')))
                    fcst_26 = fcst[fcst['ds'].dt.year == 2026]
                    fig.add_trace(go.Scatter(x=fcst_26['ds'], y=fcst_26['yhat'], name="AI Forecast", line=dict(dash='dash', color='orange')))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # --- VARIANCE & COMMENTARY (KHÔI PHỤC TẠI ĐÂY) ---
                    st.subheader("🔢 Actual vs AI Forecast Variance")
                    act_26 = m_plot[m_plot['ds'].dt.year == 2026]
                    v_df = pd.merge(act_26, fcst_26[['ds', 'yhat']], on='ds', how='inner')
                    if not v_df.empty:
                        v_df['Variance %'] = ((v_df['y'] - v_df['yhat']) / v_df['yhat']) * 100
                        st.dataframe(v_df.style.format({'ds': lambda x: x.strftime('%m/%Y'), 'y': '{:,.0f}', 'yhat': '{:,.0f}', 'Variance %': '{:+.1f}%'}), use_container_width=True)
                        
                        last_v = v_df['Variance %'].iloc[-1]
                        st.markdown("### 💡 AI Strategic Commentary")
                        if last_v > 15:
                            st.warning(f"Demand is {last_v:.1f}% higher than AI forecast. Potential IC shortage risk.")
                        elif last_v < -15:
                            st.error(f"Demand is {abs(last_v):.1f}% lower than AI forecast. Review inventory levels.")
                        else:
                            st.success("Demand is stable and aligning with AI projections.")

            with tab2:
                st.subheader("📋 2026 Strategic Plan (True Quarterly Average)")
                months_26 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
                cols_26 = [m.strftime('%m/%Y') for m in months_26]
                pivot_list = []
                last_act_date = df[df['Order qty.(A)'] > 0]['ds'].max()

                for p in top_prods:
                    product_cies = cust_df[cust_df['Material name'] == p][cie_col].unique()
                    for c in product_cies:
                        q_growth = get_quarterly_growth_logic(cust_df, p, cie_col, c)
                        f_growth = q_growth + (adj_growth/100)
                        
                        row = {'Product': p, 'CIE': str(c), 'Growth Basis': f"{f_growth*100:.1f}%"}
                        for m_date in months_26:
                            m_idx, m_str = m_date.month, m_date.strftime('%m/%Y')
                            q_idx = (m_idx - 1) // 3 + 1
                            
                            act_26 = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2026)]['Order qty.(A)'].sum()
                            
                            if act_26 > 0:
                                row[m_str] = act_26
                            elif m_date > last_act_date:
                                avg_25 = get_actual_avg_qty(cust_df, 2025, q_idx, p, cie_col, c)
                                row[m_str] = round(avg_25 * (1 + f_growth), 0)
                            else:
                                row[m_str] = 0
                        pivot_list.append(row)
                
                if pivot_list:
                    res_df = pd.DataFrame(pivot_list)
                    total_row = {'Product': 'GRAND TOTAL', 'CIE': '', 'Growth Basis': ''}
                    for col in cols_26: total_row[col] = res_df[col].sum()
                    res_df = pd.concat([res_df, pd.DataFrame([total_row])], ignore_index=True)
                    st.dataframe(res_df.style.apply(lambda x: ['background: #e6f3ff; font-weight: bold' if x['Product']=='GRAND TOTAL' else '' for _ in x], axis=1).format("{:,.0f}", subset=cols_26), use_container_width=True)
                    st.download_button("📥 Download Plan (CSV)", data=res_df.to_csv(index=False).encode('utf-8-sig'), file_name="Strategic_Plan_2026.csv")
