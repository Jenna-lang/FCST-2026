import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

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
        st.error(f"Error: {e}")
        return None

def calculate_yoy_growth(cust_df, prod_name):
    """Tính tăng trưởng dựa trên các tháng có số liệu thực tế cùng kỳ"""
    p_df = cust_df[cust_df['Material name'] == prod_name].copy()
    # Lấy các tháng đã có Actual trong 2026 (VD: T1, T2, T3)
    act_2026 = p_df[p_df['ds'].dt.year == 2026].groupby(p_df['ds'].dt.month)['Order qty.(A)'].sum()
    months_existed = act_2026.index.tolist()
    
    if not months_existed: return 0.0, 0.0 # No data in 2026 yet
    
    # Lấy dữ liệu cùng kỳ của 2025
    act_2025_same_period = p_df[(p_df['ds'].dt.year == 2025) & (p_df['ds'].dt.month.isin(months_existed))].groupby(p_df['ds'].dt.month)['Order qty.(A)'].sum()
    
    sum_26 = act_2026.sum()
    sum_25_same = act_2025_same_period.sum()
    
    # Tỷ lệ tăng trưởng thực tế (Cùng kỳ)
    growth_rate = (sum_26 - sum_25_same) / sum_25_same if sum_25_same > 0 else 0.0
    run_rate_26 = act_2026.mean() # Dùng dự phòng cho mã mới
    
    return growth_rate, run_rate_26

df = load_data()

if df is not None:
    st.sidebar.header("⚡ Settings")
    cust_col = st.sidebar.selectbox("Customer:", [c for c in df.columns if 'Customer' in c] or [df.columns[0]])
    cie_col = st.sidebar.selectbox("CIE/Color:", [c for c in df.columns if 'CIE' in c] or [df.columns[1]])
    adj_growth = st.sidebar.slider("Manual Adjustment (%)", -50, 50, 0)
    
    selected_cust = st.sidebar.selectbox("1. Select Customer:", ["-- Select --"] + sorted(df[cust_col].unique()))

    if selected_cust != "-- Select --":
        cust_df = df[df[cust_col] == selected_cust].copy()
        
        # Pareto Analysis
        rev = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        rev['Cum_Pct'] = (rev['M USD'].cumsum() / rev['M USD'].sum()) * 100
        top_prods = rev[rev['Cum_Pct'] <= 85]['Material name'].unique()[:20]
        
        tab1, tab2 = st.tabs(["📊 Digital Variance", "📋 2026 Strategic Plan"])

        with tab1:
            selected_prod = st.selectbox("Select Product:", top_prods)
            # Logic Prophet cho biểu đồ xu hướng
            p_df = cust_df[cust_df['Material name'] == selected_prod].copy()
            p_df['m'] = p_df['ds'].dt.to_period('M').dt.to_timestamp()
            df_all = p_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
            
            m = Prophet(yearly_seasonality=True).fit(df_all)
            fcst = m.predict(m.make_future_dataframe(periods=12, freq='MS'))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_all[df_all['ds'].dt.year<2026]['ds'], y=df_all['y'], name="History", line=dict(color='gray')))
            fig.add_trace(go.Scatter(x=df_all[df_all['ds'].dt.year==2026]['ds'], y=df_all['y'], name="Actual 2026", mode='markers+lines', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=fcst[fcst['ds'].dt.year==2026]['ds'], y=fcst['yhat'], name="AI Forecast", line=dict(dash='dash', color='orange')))
            st.plotly_chart(fig, use_container_width=True)
            
            # Số hóa sai lệch Q1
            st.subheader("🔢 Q1 Digital Variance")
            q1_act = df_all[df_all['ds'].dt.year == 2026]
            q1_fcst = fcst[fcst['ds'].dt.year == 2026]
            comp = pd.merge(q1_act, q1_fcst, on='ds')
            comp['Var %'] = ((comp['y'] - comp['yhat']) / comp['yhat']) * 100
            st.dataframe(comp[['ds','y','yhat','Var %']].style.format({'Var %': '{:.1f}%'}))

        with tab2:
            st.subheader("📋 Supply Plan based on YoY Growth Rate")
            months_26 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
            cols_26 = [m.strftime('%m/%Y') for m in months_26]
            
            pivot_list = []
            for p in top_prods:
                # Tính tăng trưởng dựa trên những tháng có số liệu thực tế cùng kỳ
                g_rate, r_rate = calculate_yoy_growth(cust_df, p)
                final_factor = 1 + g_rate + (adj_growth / 100)
                
                # Lấy danh sách CIE của sản phẩm này
                cies = cust_df[cust_df['Material name'] == p][cie_col].unique()
                for c in cies:
                    row = {'Product': p, 'CIE': c, 'YoY Growth': f"{g_rate*100:.1f}%"}
                    for m_date in months_26:
                        m_idx, m_str = m_date.month, m_date.strftime('%m/%Y')
                        # 1. Nếu có Actual 2026
                        act = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2026)]['Order qty.(A)'].sum()
                        if act > 0:
                            row[m_str] = act
                        elif m_idx > 3: # Dự báo tương lai
                            # 2. Ưu tiên nhân hệ số vào cùng kỳ 2025
                            h25 = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2025)]['Order qty.(A)'].sum()
                            if h25 > 0:
                                row[m_str] = round(h25 * final_factor, 0)
                            else:
                                # 3. Nếu mã mới hoàn toàn (không có h25), dùng Run-rate 2026
                                row[m_str] = round(r_rate * final_factor, 0)
                        else: row[m_str] = 0
                    pivot_list.append(row)
            
            res_df = pd.DataFrame(pivot_list)
            # Grand Total
            total_row = {'Product': 'GRAND TOTAL', 'CIE': '---', 'YoY Growth': '---'}
            for col in cols_26: total_row[col] = res_df[col].sum()
            res_df = pd.concat([res_df, pd.DataFrame([total_row])], ignore_index=True)
            
            st.dataframe(res_df.style.apply(lambda x: ['font-weight:bold; background:#e6f3ff' if x['Product']=='GRAND TOTAL' else '' for _ in x], axis=1).format("{:,.0f}", subset=cols_26), use_container_width=True)
