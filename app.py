import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. System Setup
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

def calculate_yoy_growth(cust_df, prod_name):
    """Tính tăng trưởng dựa trên dữ liệu cùng kỳ (Year-over-Year)"""
    p_df = cust_df[cust_df['Material name'] == prod_name].copy()
    # Lấy các tháng đã có số liệu thực tế trong 2026
    act_2026 = p_df[p_df['ds'].dt.year == 2026].groupby(p_df['ds'].dt.month)['Order qty.(A)'].sum()
    active_months = act_2026.index.tolist()
    
    if not active_months: return 0.0, 0.0
    
    # Lấy dữ liệu cùng kỳ của năm 2025
    act_2025_same = p_df[(p_df['ds'].dt.year == 2025) & (p_df['ds'].dt.month.isin(active_months))].groupby(p_df['ds'].dt.month)['Order qty.(A)'].sum()
    
    yoy_growth = (act_2026.sum() - act_2025_same.sum()) / act_2025_same.sum() if act_2025_same.sum() > 0 else 0.0
    run_rate = act_2026.mean() 
    
    return yoy_growth, run_rate

df = load_data()

if df is not None:
    st.sidebar.header("⚡ Settings")
    cust_col = st.sidebar.selectbox("Customer:", [c for c in df.columns if 'Customer' in c] or [df.columns[0]])
    cie_col = st.sidebar.selectbox("CIE/Color:", [c for c in df.columns if 'CIE' in c] or [df.columns[1]])
    adj_growth = st.sidebar.slider("Manual Adjustment (%)", -50, 50, 0)
    
    selected_cust = st.sidebar.selectbox("1. Select Customer:", ["-- Select --"] + sorted(df[cust_col].unique()))

    if selected_cust != "-- Select --":
        cust_df = df[df[cust_col] == selected_cust].copy()
        
        # Pareto 85% Logic
        rev = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        rev['Cum_Pct'] = (rev['M USD'].cumsum() / rev['M USD'].sum()) * 100
        top_prods = rev[rev['Cum_Pct'] <= 85]['Material name'].unique()[:20]

        tab1, tab2 = st.tabs(["📊 Performance & AI Audit", "📋 2026 Strategic Plan"])

        with tab1:
            selected_prod = st.selectbox("Product:", top_prods)
            yoy_g, r_rate = calculate_yoy_growth(cust_df, selected_prod)
            
            p_df = cust_df[cust_df['Material name'] == selected_prod].copy()
            p_df['m'] = p_df['ds'].dt.to_period('M').dt.to_timestamp()
            df_plot = p_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
            
            # AI Trend Analysis (Dựa trên toàn bộ lịch sử các năm qua)
            m = Prophet(yearly_seasonality=True).fit(df_plot)
            future = m.make_future_dataframe(periods=12, freq='MS')
            fcst = m.predict(future)
            
            # --- CHART: ACTUAL (REAL) VS AI (TREND) ---
            fig = go.Figure()
            # Actual 2026 (Xanh dương đậm)
            act_2026 = df_plot[df_plot['ds'].dt.year == 2026]
            fig.add_trace(go.Scatter(x=act_2026['ds'], y=act_2026['y'], name="Actual 2026", mode='lines+markers', line=dict(color='blue', width=4)))
            
            # AI Forecast Trend (Cam đứt đoạn)
            fcst_2026 = fcst[fcst['ds'].dt.year == 2026]
            fig.add_trace(go.Scatter(x=fcst_2026['ds'], y=fcst_2026['yhat'], name="AI Trend Forecast", line=dict(dash='dash', color='orange', width=2)))
            
            # Historical History (Mờ)
            fig.add_trace(go.Scatter(x=df_plot[df_plot['ds'].dt.year < 2026]['ds'], y=df_plot['y'], name="History", line=dict(color='lightgray')))

            fig.update_layout(title=f"Accuracy Audit: {selected_prod} (YoY Growth: {yoy_g*100:.1f}%)", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # --- VARIANCE TABLE (RED ALERT) ---
            st.subheader("🔢 Digital Variance Analysis")
            comp = pd.merge(act_2026, fcst[['ds', 'yhat']], on='ds')
            if not comp.empty:
                comp['Variance %'] = ((comp['y'] - comp['yhat']) / comp['yhat']) * 100
                st.dataframe(comp[['ds','y','yhat','Variance %']].style.format({
                    'ds': lambda x: x.strftime('%m/%Y'), 'y': '{:,.0f}', 'yhat': '{:,.0f}', 'Variance %': '{:.1f}%'
                }).applymap(lambda x: 'color: red; font-weight: bold' if abs(x) > 20 else 'color: green', subset=['Variance %']), use_container_width=True)

        with tab2:
            st.subheader("📋 2026 Strategic Plan (YoY Growth Applied)")
            # [Logic Tab 2 nhân hệ số tăng trưởng chuẩn như đã thống nhất]
            months_26 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
            cols_26 = [m.strftime('%m/%Y') for m in months_26]
            pivot_list = []
            for p in top_prods:
                y_g, r_r = calculate_yoy_growth(cust_df, p)
                final_f = 1 + y_g + (adj_growth / 100)
                cies = cust_df[cust_df['Material name'] == p][cie_col].unique()
                for c in cies:
                    row = {'Product': p, 'CIE': c, 'YoY Growth': f"{y_g*100:.1f}%"}
                    for m_date in months_26:
                        m_idx, m_str = m_date.month, m_date.strftime('%m/%Y')
                        act = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2026)]['Order qty.(A)'].sum()
                        if act > 0: row[m_str] = act
                        elif m_idx > 3:
                            h25 = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2025)]['Order qty.(A)'].sum()
                            row[m_str] = round(h25 * final_f, 0) if h25 > 0 else round(r_r * final_f, 0)
                        else: row[m_str] = 0
                    pivot_list.append(row)
            res_df = pd.DataFrame(pivot_list)
            total_row = {'Product': 'GRAND TOTAL', 'CIE': '---', 'YoY Growth': '---'}
            for col in cols_26: total_row[col] = res_df[col].sum()
            res_df = pd.concat([res_df, pd.DataFrame([total_row])], ignore_index=True)
            st.dataframe(res_df.style.apply(lambda x: ['font-weight:bold; background:#e6f3ff' if x['Product']=='GRAND TOTAL' else '' for _ in x], axis=1).format("{:,.0f}", subset=cols_26), use_container_width=True)
