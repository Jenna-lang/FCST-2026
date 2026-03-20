import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. Cấu hình hệ thống
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
        st.error(f"Error loading file: {e}")
        return None

@st.cache_data(ttl=3600)
def get_full_analysis(cust_df, prod_name):
    p_df = cust_df[cust_df['Material name'] == prod_name].copy()
    if len(p_df) < 5: return None, None, 0.0
    
    p_df['m'] = p_df['ds'].dt.to_period('M').dt.to_timestamp()
    df_all = p_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
    
    # Huấn luyện AI trên toàn bộ dữ liệu để có dự báo tốt nhất cho tương lai
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_all)
    future = m.make_future_dataframe(periods=12, freq='MS')
    fcst = m.predict(future)
    
    # Tính toán tăng trưởng AI (2026 vs 2025)
    t25 = df_all[df_all['ds'].dt.year == 2025]['y'].sum()
    a26 = df_all[df_all['ds'].dt.year == 2026]['y'].sum()
    f26_future = fcst[(fcst['ds'].dt.year == 2026) & (fcst['ds'] > df_all['ds'].max())]['yhat'].sum()
    growth = ((a26 + f26_future - t25) / t25) if t25 > 0 else 0.0
    
    return df_all, fcst[['ds', 'yhat']], growth

df = load_data()

if df is not None:
    st.sidebar.header("⚡ Settings")
    cust_col = st.sidebar.selectbox("Customer:", [c for c in df.columns if 'Customer' in c] or [df.columns[0]])
    cie_col = st.sidebar.selectbox("CIE/Color:", [c for c in df.columns if 'CIE' in c] or [df.columns[1]])
    adj_growth = st.sidebar.slider("AI Adjustment (%)", -50, 50, 0)
    
    cust_list = sorted(df[cust_col].unique())
    selected_cust = st.sidebar.selectbox("1. Select Customer:", ["-- Select --"] + cust_list)

    if selected_cust != "-- Select --":
        cust_df = df[df[cust_col] == selected_cust].copy()
        sales = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        top_prods = sales['Material name'].unique()[:20]
        
        tab1, tab2 = st.tabs(["📊 Digital Analysis & Trend", "📋 2026 Production Plan"])

        with tab1:
            selected_prod = st.selectbox("Select Product:", top_prods)
            actuals_all, fcst_res, ai_g = get_full_analysis(cust_df, selected_prod)
            
            if fcst_res is not None:
                # --- PHẦN 1: BIỂU ĐỒ TỔNG THỂ (LOGIC CŨ) ---
                fig = go.Figure()
                # Quá khứ
                fig.add_trace(go.Scatter(x=actuals_all[actuals_all['ds'] < '2026-01-01']['ds'], y=actuals_all[actuals_all['ds'] < '2026-01-01']['y'], name="History", line=dict(color='gray', width=1)))
                # Thực tế 2026
                fig.add_trace(go.Scatter(x=actuals_all[actuals_all['ds'] >= '2026-01-01']['ds'], y=actuals_all[actuals_all['ds'] >= '2026-01-01']['y'], name="Actual 2026", mode='lines+markers', line=dict(color='blue', width=3)))
                # Dự báo AI
                fig.add_trace(go.Scatter(x=fcst_res[fcst_res['ds'] >= '2026-01-01']['ds'], y=fcst_res[fcst_res['ds'] >= '2026-01-01']['yhat'], name="AI Forecast", line=dict(color='orange', dash='dash')))
                
                fig.update_layout(title=f"Trend Analysis: {selected_prod}", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

                # --- PHẦN 2: SỐ HÓA SAI LỆCH (LOGIC MỚI) ---
                st.subheader("🔢 Digital Variance Check (Jan - Mar 2026)")
                act_q1 = actuals_all[(actuals_all['ds'] >= '2026-01-01') & (actuals_all['ds'] <= '2026-03-01')]
                fcst_q1 = fcst_res[(fcst_res['ds'] >= '2026-01-01') & (fcst_res['ds'] <= '2026-03-01')]
                
                comp_df = pd.merge(act_q1, fcst_q1, on='ds', how='inner')
                if not comp_df.empty:
                    comp_df['Diff'] = comp_df['y'] - comp_df['yhat']
                    comp_df['Variance_%'] = (comp_df['Diff'] / comp_df['yhat']) * 100
                    
                    st.dataframe(comp_df.style.format({
                        'ds': lambda x: x.strftime('%m/%Y'),
                        'y': '{:,.0f}', 'yhat': '{:,.0f}', 'Diff': '{:,.0f}', 'Variance_%': '{:.1f}%'
                    }).applymap(lambda x: 'color: red; font-weight: bold' if abs(x) > 20 else 'color: green', subset=['Variance_%']), use_container_width=True)
                else:
                    st.info("No Actual data in 2026 to compare yet.")

        with tab2:
            st.subheader("📋 2026 Supply Plan")
            with st.spinner("Calculating Plan..."):
                growth_map = {p: get_full_analysis(cust_df, p)[2] for p in top_prods}
            
            # Logic tính toán phân bổ tháng cho Tab 2 (Giữ nguyên như các bản trước)
            # ... (Phần code tính pivot_list cho Plan) ...
            months_26 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
            cols_26 = [m.strftime('%m/%Y') for m in months_26]
            all_items = cust_df[cust_df['Material name'].isin(top_prods)][['Material name', cie_col]].drop_duplicates()
            pivot_list = []
            
            for _, r in all_items.iterrows():
                p, c = r['Material name'], r[cie_col]
                factor = 1 + growth_map.get(p, 0.0) + (adj_growth / 100)
                row = {'Product': p, 'CIE': c, 'Growth': f"{(factor-1)*100:.1f}%"}
                for m_date in months_26:
                    m_idx, m_str = m_date.month, m_date.strftime('%m/%Y')
                    # Lấy Actual nếu có, không thì lấy Dự báo
                    actual_val = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2026)]['Order qty.(A)'].sum()
                    if actual_val > 0: row[m_str] = actual_val
                    else:
                        hist_25 = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2025)]['Order qty.(A)'].sum()
                        row[m_str] = round(hist_25 * factor, 0)
                pivot_list.append(row)
            
            res_df = pd.DataFrame(pivot_list)
            # Dòng Total
            total_row = {'Product': 'GRAND TOTAL', 'CIE': '---', 'Growth': '---'}
            for col in cols_26: total_row[col] = res_df[col].sum()
            res_df = pd.concat([res_df, pd.DataFrame([total_row])], ignore_index=True)
            st.dataframe(res_df.style.apply(lambda x: ['font-weight:bold; background:#e6f3ff' if x['Product']=='GRAND TOTAL' else '' for _ in x], axis=1).format("{:,.0f}", subset=cols_26), use_container_width=True)
