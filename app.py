import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. System Setup
st.set_page_config(page_title="AI Supply Chain Advisor 2026", layout="wide")

def get_quarterly_growth(cust_df, prod_name, cie_col, cie_val):
    """Tính tăng trưởng dựa trên trung bình các tháng ĐÃ CÓ số liệu trong quý."""
    p_df = cust_df[(cust_df['Material name'] == prod_name) & (cust_df[cie_col] == cie_val)].copy()
    if p_df.empty: return 0.0
    
    p_df['quarter'] = p_df['ds'].dt.quarter
    p_df['year'] = p_df['ds'].dt.year
    
    # Chỉ lấy những dòng có số lượng thực tế > 0 để tính trung bình
    actual_data = p_df[p_df['Order qty.(A)'] > 0]
    if actual_data.empty: return 0.0

    # Gom nhóm và tính trung bình (Pandas .mean() tự động bỏ qua các tháng không có số liệu)
    q_avg = actual_data.groupby(['year', 'quarter'])['Order qty.(A)'].mean().reset_index()
    
    current_26 = q_avg[q_avg['year'] == 2026]
    if current_26.empty: return 0.0
    
    # Lấy quý gần nhất của 2026 đã có số
    latest_q = current_26['quarter'].max()
    avg_26 = current_26[current_26['quarter'] == latest_q]['Order qty.(A)'].values[0]
    
    # So sánh với trung bình cùng quý đó năm 2025
    avg_25_data = q_avg[(q_avg['year'] == 2025) & (q_avg['quarter'] == latest_q)]['Order qty.(A)'].values
    
    if len(avg_25_data) > 0 and avg_25_data[0] > 0:
        growth = (avg_26 / avg_25_data[0]) - 1
        return min(max(growth, -0.5), 0.5) # Giới hạn biên độ +/- 50%
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

# --- UI ---
st.sidebar.header("📁 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload AICheck.xlsx", type=['xlsx'])

if uploaded_file:
    df = process_data(uploaded_file)
    if df is not None:
        all_cols = df.columns.tolist()
        cust_col = next((c for c in all_cols if 'customer' in c.lower()), all_cols[0])
        cie_col = next((c for c in all_cols if c not in [cust_col, 'ds', 'Material name', 'Order qty.(A)', 'M USD'] and 'date' not in c.lower()), all_cols[1])
        
        adj_growth = st.sidebar.slider("Global Adjustment (%)", -50, 50, 0)
        selected_cust = st.sidebar.selectbox("Select Customer:", ["-- Select --"] + sorted(df[cust_col].unique().tolist()))

        if selected_cust != "-- Select --":
            cust_df = df[df[cust_col] == selected_cust].copy()
            # Pareto 85%
            rev = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
            top_prods = rev[rev['M USD'].cumsum() / rev['M USD'].sum() <= 0.86]['Material name'].unique()[:20]

            tab1, tab2 = st.tabs(["📊 Performance Audit", "📋 2026 Strategic Plan"])

            with tab2:
                st.subheader("📋 2026 Strategic Plan (True Quarterly Average)")
                months_26 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
                cols_26 = [m.strftime('%m/%Y') for m in months_26]
                pivot_list = []
                last_act_date = df[df['Order qty.(A)'] > 0]['ds'].max()

                for p in top_prods:
                    for c in cust_df[cust_df['Material name'] == p][cie_col].unique():
                        q_growth = get_quarterly_growth(cust_df, p, cie_col, c)
                        f_growth = q_growth + (adj_growth/100)
                        
                        row = {'Product': p, 'CIE': c, 'Growth Basis': f"{f_growth*100:.1f}%"}
                        
                        for m_date in months_26:
                            m_idx, m_str = m_date.month, m_date.strftime('%m/%Y')
                            q_idx = (m_idx - 1) // 3 + 1
                            
                            act_26 = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2026)]['Order qty.(A)'].sum()
                            
                            if act_26 > 0:
                                row[m_str] = act_26
                            elif m_date > last_act_date:
                                # Lấy trung bình số lượng các tháng có số của quý tương ứng năm 2025
                                hist_25 = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & (cust_df['ds'].dt.year==2025) & (cust_df['ds'].dt.quarter == q_idx)]
                                avg_hist_25 = hist_25[hist_25['Order qty.(A)'] > 0]['Order qty.(A)'].mean()
                                
                                base_val = avg_hist_25 if (pd.notna(avg_hist_25) and avg_hist_25 > 0) else 0
                                row[m_str] = round(base_val * (1 + f_growth), 0)
                            else:
                                row[m_str] = 0
                        pivot_list.append(row)
                
                if pivot_list:
                    res_df = pd.DataFrame(pivot_list)
                    total_row = {'Product': 'GRAND TOTAL', 'CIE': '---', 'Growth Basis': '---'}
                    for col in cols_26: total_row[col] = res_df[col].sum()
                    res_df = pd.concat([res_df, pd.DataFrame([total_row])], ignore_index=True)
                    st.dataframe(res_df.style.apply(lambda x: ['background: #e6f3ff; font-weight: bold' if x['Product']=='GRAND TOTAL' else '' for _ in x], axis=1).format("{:,.0f}", subset=cols_26), use_container_width=True)
