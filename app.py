import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. System Configuration
st.set_page_config(page_title="AI Supply Chain Advisor 2026", layout="wide")

# --- HÀM XỬ LÝ DỮ LIỆU ---
def process_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = [str(col).strip() for col in df.columns]
        date_col = 'Requested deliv. date'
        if date_col in df.columns:
            df['ds'] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=['ds'])
            return df
        else:
            st.error(f"Thiếu cột '{date_col}' trong file!")
            return None
    except Exception as e:
        st.error(f"Lỗi định dạng file: {e}")
        return None

def calculate_metrics(cust_df, prod_name):
    p_df = cust_df[cust_df['Material name'] == prod_name].copy()
    
    # Tính Monthly Average Growth Rate (Tăng trưởng trung bình hàng tháng)
    p_df['m_idx'] = p_df['ds'].dt.to_period('M').dt.to_timestamp()
    monthly_sum = p_df.groupby('m_idx')['Order qty.(A)'].sum().reset_index()
    monthly_sum['Pct_Change'] = monthly_sum['Order qty.(A)'].pct_change() * 100
    avg_growth = monthly_sum['Pct_Change'].mean()
    
    # Tính YoY Growth (Logic cũ của bạn)
    act_2026 = p_df[p_df['ds'].dt.year == 2026].groupby(p_df['ds'].dt.month)['Order qty.(A)'].sum()
    active_months = act_2026.index.tolist()
    if not active_months: return 0.0, 0.0, 0.0
    
    act_2025_same = p_df[(p_df['ds'].dt.year == 2025) & (p_df['ds'].dt.month.isin(active_months))].groupby(p_df['ds'].dt.month)['Order qty.(A)'].sum()
    yoy_growth = (act_2026.sum() - act_2025_same.sum()) / act_2025_same.sum() if act_2025_same.sum() > 0 else 0.0
    
    return avg_growth, yoy_growth, act_2026.mean()

# --- SIDEBAR: FILE UPLOADER ---
st.sidebar.header("📁 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload AICheck.xlsx file", type=['xlsx'])

if uploaded_file is not None:
    df = process_data(uploaded_file)
    
    if df is not None:
        st.sidebar.header("⚡ Analysis Settings")
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
            if len(top_prods) == 0:
                top_prods = cust_df.groupby('Material name')['Order qty.(A)'].sum().nlargest(10).index

            tab1, tab2 = st.tabs(["📊 Performance & AI Audit", "📋 2026 Strategic Plan"])

            with tab1:
                selected_prod = st.selectbox("Product Audit:", top_prods)
                
                # Gọi hàm tính toán metrics mới
                avg_g, yoy_g, r_rate = calculate_metrics(cust_df, selected_prod)
                
                # Hiển thị Metrics (Tỷ lệ tăng trưởng trung bình được thêm ở đây)
                m1, m2, m3 = st.columns(3)
                m1.metric("Avg Monthly Growth", f"{avg_g:.1f}%")
                m2.metric("YoY Growth (26 vs 25)", f"{yoy_g*100:.1f}%")
                m3.metric("Run-rate (Avg 2026)", f"{r_rate:,.0f}")

                # --- CHART ACTUAL VS AI TREND ---
                p_df = cust_df[cust_df['Material name'] == selected_prod].copy()
                p_df['m'] = p_df['ds'].dt.to_period('M').dt.to_timestamp()
                df_plot = p_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
                
                m = Prophet(yearly_seasonality=True).fit(df_plot)
                future = m.make_future_dataframe(periods=12, freq='MS')
                fcst = m.predict(future)
                
                fig = go.Figure()
                act_2026_p = df_plot[df_plot['ds'].dt.year == 2026]
                fig.add_trace(go.Scatter(x=act_2026_p['ds'], y=act_2026_p['y'], name="Actual 2026", mode='lines+markers', line=dict(color='blue', width=4)))
                fcst_2026_p = fcst[fcst['ds'].dt.year == 2026]
                fig.add_trace(go.Scatter(x=fcst_2026_p['ds'], y=fcst_2026_p['yhat'], name="AI Trend", line=dict(dash='dash', color='orange')))
                fig.add_trace(go.Scatter(x=df_plot[df_plot['ds'].dt.year < 2026]['ds'], y=df_plot['y'], name="History", line=dict(color='lightgray')))
                fig.update_layout(title=f"Audit: {selected_prod}", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

                # --- VARIANCE & YTD ---
                comp = pd.merge(act_2026_p, fcst[['ds', 'yhat']], on='ds')
                if not comp.empty:
                    comp['Var %'] = ((comp['y'] - comp['yhat']) / comp['yhat']) * 100
                    comp['YTD Var %'] = ((comp['y'].cumsum() - comp['yhat'].cumsum()) / comp['yhat'].cumsum()) * 100
                    st.dataframe(comp[['ds','y','yhat','Var %', 'YTD Var %']].style.format({
                        'ds': lambda x: x.strftime('%m/%Y'), 'y': '{:,.0f}', 'yhat': '{:,.0f}', 'Var %': '{:.1f}%', 'YTD Var %': '{:.1f}%'
                    }).applymap(lambda x: 'color: red; font-weight: bold' if abs(x) > 20 else 'color: green', subset=['Var %', 'YTD Var %']), use_container_width=True)

                    # AI Comment
                    st.markdown("### 💬 AI Analytical Comment")
                    last_ytd = comp['YTD Var %'].iloc[-1]
                    status = "STABLE" if abs(last_ytd) <= 10 else ("SURGE" if last_ytd > 10 else "DROP")
                    st.info(f"**Status:** {status} | **Advice:** Review your {status.lower()} strategy for the next S&OP meeting.")

            with tab2:
                st.subheader("📋 2026 Strategic Plan")
                months_26 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
                cols_26 = [m.strftime('%m/%Y') for m in months_26]
                pivot_list = []
                for p in top_prods:
                    # Dùng lại hàm calculate_metrics để lấy YoY và Run-rate
                    _, y_g_v, r_r_v = calculate_metrics(cust_df, p)
                    final_f = 1 + y_g_v + (adj_growth / 100)
                    cies = cust_df[cust_df['Material name'] == p][cie_col].unique()
                    for c in cies:
                        row = {'Product': p, 'CIE': c, 'YoY Growth': f"{y_g_v*100:.1f}%"}
                        for m_date in months_26:
                            m_idx, m_str = m_date.month, m_date.strftime('%m/%Y')
                            act_v = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2026)]['Order qty.(A)'].sum()
                            if act_v > 0: row[m_str] = act_v
                            elif m_date > (act_2026_p['ds'].max() if not act_2026_p.empty else pd.Timestamp('2026-03-01')):
                                h25 = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2025)]['Order qty.(A)'].sum()
                                row[m_str] = round(h25 * final_f, 0) if h25 > 0 else round(r_r_v * final_f, 0)
                            else: row[m_str] = 0
                        pivot_list.append(row)
                
                if pivot_list:
                    res_df = pd.DataFrame(pivot_list)
                    total_row = {'Product': 'GRAND TOTAL', 'CIE': '---', 'YoY Growth': '---'}
                    for col in cols_26: total_row[col] = res_df[col].sum()
                    res_df = pd.concat([res_df, pd.DataFrame([total_row])], ignore_index=True)
                    st.dataframe(res_df.style.format("{:,.0f}", subset=cols_26), use_container_width=True)
                    st.download_button("📥 Export Plan", res_df.to_csv(index=False).encode('utf-8-sig'), "Strategic_Plan_2026.csv")

else:
    st.info("👋 Chào Jenna! Vui lòng upload file 'AICheck.xlsx' ở thanh bên trái để bắt đầu phân tích.")
