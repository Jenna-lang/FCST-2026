import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. Cấu hình hệ thống
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
    # Lọc dữ liệu sản phẩm cụ thể
    p_df = cust_df[cust_df['Material name'] == prod_name].copy()
    
    # 1. Tính Average Monthly Growth (Gom nhóm tháng trước khi tính % thay đổi)
    monthly_all = p_df.groupby(p_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
    monthly_all['Pct_Change'] = monthly_all['Order qty.(A)'].pct_change() * 100
    avg_growth = monthly_all['Pct_Change'].mean()
    
    # 2. Tính số thực tế 2026 (Chỉ lấy các dòng thuộc năm 2026)
    df_2026 = p_df[p_df['ds'].dt.year == 2026].copy()
    act_2026_monthly = df_2026.groupby(df_2026['ds'].dt.month)['Order qty.(A)'].sum()
    
    # 3. Tính YoY Growth (So sánh phần đã qua của 2026 với đúng các tháng đó của 2025)
    active_months = act_2026_monthly.index.tolist()
    if not active_months: return avg_growth, 0.0, 0.0
    
    df_2025_same = p_df[(p_df['ds'].dt.year == 2025) & (p_df['ds'].dt.month.isin(active_months))]
    act_2025_sum = df_2025_same['Order qty.(A)'].sum()
    act_2026_sum = act_2026_monthly.sum()
    
    yoy_growth = (act_2026_sum - act_2025_sum) / act_2025_sum if act_2025_sum > 0 else 0.0
    
    return avg_growth, yoy_growth, act_2026_monthly.mean()

# --- SIDEBAR: QUẢN LÝ FILE ---
st.sidebar.header("📁 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload AICheck.xlsx file", type=['xlsx'])

if uploaded_file is not None:
    df = process_data(uploaded_file)
    if df is not None:
        st.sidebar.header("⚡ Analysis Settings")
        cust_col = st.sidebar.selectbox("Customer Column:", [c for c in df.columns if 'Customer' in c] or [df.columns[0]])
        cie_col = st.sidebar.selectbox("CIE/Color Column:", [c for c in df.columns if 'CIE' in c] or [df.columns[1]])
        adj_growth = st.sidebar.slider("Manual Adjustment (%)", -50, 50, 0)
        selected_cust = st.sidebar.selectbox("1. Select Customer:", ["-- Select --"] + sorted(df[cust_col].unique()))

        if selected_cust != "-- Select --":
            # Lọc dữ liệu theo khách hàng ngay từ đầu để đảm bảo số liệu sạch
            cust_df = df[df[cust_col] == selected_cust].copy()
            
            # Pareto 85% Logic
            rev = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
            rev['Cum_Pct'] = (rev['M USD'].cumsum() / rev['M USD'].sum()) * 100
            top_prods = rev[rev['Cum_Pct'] <= 86]['Material name'].unique()[:20]

            tab1, tab2 = st.tabs(["📊 Performance & AI Audit", "📋 2026 Strategic Plan"])

            with tab1:
                selected_prod = st.selectbox("Product Audit:", top_prods)
                avg_g, yoy_g, r_rate = calculate_metrics(cust_df, selected_prod)
                
                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Avg Monthly Growth", f"{avg_g:.1f}%")
                m2.metric("YoY Growth (26 vs 25)", f"{yoy_g*100:.1f}%")
                m3.metric("Run-rate (Avg 2026)", f"{r_rate:,.0f}")

                # Plotting
                p_df = cust_df[cust_df['Material name'] == selected_prod].copy()
                df_plot = p_df.groupby(p_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
                df_plot['ds'] = df_plot['ds'].dt.to_timestamp()
                df_plot = df_plot.rename(columns={'Order qty.(A)': 'y'})
                
                m = Prophet(yearly_seasonality=True).fit(df_plot)
                future = m.make_future_dataframe(periods=12, freq='MS')
                fcst = m.predict(future)
                
                fig = go.Figure()
                act_26 = df_plot[df_plot['ds'].dt.year == 2026]
                fig.add_trace(go.Scatter(x=act_26['ds'], y=act_26['y'], name="Actual 2026", mode='lines+markers', line=dict(color='blue', width=4)))
                fcst_26 = fcst[fcst['ds'].dt.year == 2026]
                fig.add_trace(go.Scatter(x=fcst_26['ds'], y=fcst_26['yhat'], name="AI Trend", line=dict(dash='dash', color='orange')))
                st.plotly_chart(fig, use_container_width=True)

                # Variance Table (Bảng này sẽ giúp Jenna đối soát 3 tháng đầu năm)
                comp = pd.merge(act_26, fcst[['ds', 'yhat']], on='ds')
                if not comp.empty:
                    comp['Var %'] = ((comp['y'] - comp['yhat']) / comp['yhat']) * 100
                    st.subheader("🔢 Chi tiết thực tế & Dự báo 2026")
                    st.dataframe(comp[['ds','y','yhat','Var %']].style.format({
                        'ds': lambda x: x.strftime('%m/%Y'), 'y': '{:,.0f}', 'yhat': '{:,.0f}', 'Var %': '{:.1f}%'
                    }), use_container_width=True)

            with tab2:
                st.subheader("📋 2026 Strategic Plan")
                months_26 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
                cols_26 = [m.strftime('%m/%Y') for m in months_26]
                pivot_list = []
                for p in top_prods:
                    _, y_g_v, r_r_v = calculate_metrics(cust_df, p)
                    final_f = 1 + y_g_v + (adj_growth / 100)
                    cies = cust_df[cust_df['Material name'] == p][cie_col].unique()
                    for c in cies:
                        row = {'Product': p, 'CIE': c, 'YoY Growth': f"{y_g_v*100:.1f}%"}
                        for m_date in months_26:
                            m_idx, m_str = m_date.month, m_date.strftime('%m/%Y')
                            # Lọc chính xác theo Product + CIE + Tháng + Năm
                            act_v = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & 
                                            (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2026)]['Order qty.(A)'].sum()
                            
                            if act_v > 0:
                                row[m_str] = act_v
                            elif m_date > act_26['ds'].max():
                                h25 = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & 
                                              (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2025)]['Order qty.(A)'].sum()
                                row[m_str] = round(h25 * final_f, 0) if h25 > 0 else round(r_r_v * final_f, 0)
                            else:
                                row[m_str] = 0
                        pivot_list.append(row)
                
                if pivot_list:
                    res_df = pd.DataFrame(pivot_list)
                    total_row = {'Product': 'GRAND TOTAL', 'CIE': '---', 'YoY Growth': '---'}
                    for col in cols_26: total_row[col] = res_df[col].sum()
                    res_df = pd.concat([res_df, pd.DataFrame([total_row])], ignore_index=True)
                    st.dataframe(res_df.style.apply(lambda x: ['font-weight:bold; background:#e6f3ff' if x['Product']=='GRAND TOTAL' else '' for _ in x], axis=1).format("{:,.0f}", subset=cols_26), use_container_width=True)
                    st.download_button("📥 Export CSV", res_df.to_csv(index=False).encode('utf-8-sig'), "Strategic_Plan_2026.csv")
else:
    st.info("👋 Chào Jenna! Hãy upload file 'AICheck.xlsx' để bắt đầu.")
