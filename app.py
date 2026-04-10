import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. System Setup
st.set_page_config(page_title="AI Supply Chain Advisor 2026", layout="wide")

# Khởi tạo bộ nhớ để lưu Adjustment riêng cho từng sản phẩm
if 'adj_dict' not in st.session_state:
    st.session_state.adj_dict = {}

# --- CORE LOGIC FUNCTIONS ---

def get_actual_avg_qty(df, year, quarter, prod, cie_col, cie_val, mat_col):
    """Tính trung bình thực tế tháng > 0 trong một Quý."""
    temp = df[(df[mat_col] == prod) & 
              (df[cie_col] == cie_val) & 
              (df['ds'].dt.year == year) & 
              (df['ds'].dt.quarter == quarter)].copy()
    
    monthly_sum = temp.groupby(temp['ds'].dt.month)['Order qty.(A)'].sum()
    actual_months = monthly_sum[monthly_sum > 0]
    
    if actual_months.empty:
        return 0.0
    return actual_months.mean()

def get_quarterly_growth_logic(cust_df, prod, cie_col, cie_val, mat_col):
    """So sánh TB quý gần nhất 2026 vs cùng kỳ 2025."""
    df_26 = cust_df[cust_df['ds'].dt.year == 2026].copy()
    if df_26.empty: return 0.0
    
    valid_26 = df_26[df_26['Order qty.(A)'] > 0]
    if valid_26.empty: return 0.0
    
    latest_q_26 = valid_26['ds'].dt.quarter.max()
    
    avg_26 = get_actual_avg_qty(cust_df, 2026, latest_q_26, prod, cie_col, cie_val, mat_col)
    avg_25 = get_actual_avg_qty(cust_df, 2025, latest_q_26, prod, cie_col, cie_val, mat_col)
    
    if avg_25 > 0:
        growth = (avg_26 / avg_25) - 1
        return min(max(growth, -0.5), 0.5)
    return 0.0

def process_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = [str(col).strip() for col in df.columns]
        date_col = next((c for c in df.columns if 'date' in c.lower()), None)
        qty_col = next((c for c in df.columns if 'qty' in c.lower()), None)
        if date_col and qty_col:
            df['ds'] = pd.to_datetime(df[date_col], errors='coerce')
            df['Order qty.(A)'] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
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
        cust_col = st.sidebar.selectbox("Identify 'Customer' Column:", all_cols, index=0)
        cie_col = st.sidebar.selectbox("Identify 'CIE / Item Code' Column:", all_cols, index=1)
        mat_col = next((c for c in all_cols if 'material' in c.lower()), all_cols[0])
        
        selected_cust = st.sidebar.selectbox("Select Target Customer:", ["-- Select --"] + sorted(df[cust_col].unique().tolist()))

        if selected_cust != "-- Select --":
            cust_df = df[df[cust_col] == selected_cust].copy()
            rev_col = next((c for c in all_cols if 'm usd' in c.lower()), 'Order qty.(A)')
            rev = cust_df.groupby(mat_col)[rev_col].sum().sort_values(ascending=False).reset_index()
            top_prods = rev[rev[rev_col].cumsum() / rev[rev_col].sum() <= 0.86][mat_col].unique()[:30]

            tab1, tab2 = st.tabs(["📊 Performance Audit & Adjust", "📋 2026 Strategic Plan"])

            with tab1:
                st.subheader("🔍 Individual Product Tuning")
                selected_prod = st.selectbox("Select Product to Audit:", top_prods)
                
                # --- LOGIC ADJUSTMENT RIÊNG BIỆT ---
                current_adj = st.session_state.adj_dict.get(selected_prod, 0)
                new_adj = st.number_input(f"Adjustment for {selected_prod} (%)", value=current_adj, step=5)
                st.session_state.adj_dict[selected_prod] = new_adj # Lưu vào bộ nhớ
                
                sample_cies = cust_df[cust_df[mat_col] == selected_prod][cie_col].unique()
                q_growth = get_quarterly_growth_logic(cust_df, selected_prod, cie_col, sample_cies[0], mat_col)
                f_growth = q_growth + (new_adj/100)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Quarterly Growth", f"{q_growth*100:.1f}%")
                m2.metric("Final (Incl. Adj)", f"{f_growth*100:.1f}%")
                m3.metric("Status", "Balanced" if abs(new_adj) < 20 else "Aggressive")

                # Bảng so sánh
                p_plot = cust_df[cust_df[mat_col] == selected_prod].groupby(cust_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
                p_plot['ds'] = p_plot['ds'].dt.to_timestamp()
                p_plot = p_plot.rename(columns={'Order qty.(A)': 'y'})
                
                if len(p_plot) > 2:
                    st.subheader("🔢 Actual vs AI Variance")
                    act_26 = p_plot[p_plot['ds'].dt.year == 2026]
                    # Dự báo tạm thời để audit
                    model = Prophet().fit(p_plot)
                    fcst = model.predict(model.make_future_dataframe(periods=6, freq='MS'))
                    fcst_26 = fcst[fcst['ds'].dt.year == 2026]
                    
                    v_df = pd.merge(act_26, fcst_26[['ds', 'yhat']], on='ds', how='inner')
                    if not v_df.empty:
                        v_df['Variance %'] = ((v_df['y'] - v_df['yhat']) / v_df['yhat']) * 100
                        avg_v = v_df['Variance %'].mean()
                        st.write(f"**Average Variance:** {avg_v:.1f}% | **Recommended Adj:** {avg_v-20:.0f}%" if avg_v > 20 else f"**Average Variance:** {avg_v:.1f}%")
                        st.dataframe(v_df.style.format({'y': '{:,.0f}', 'yhat': '{:,.0f}', 'Variance %': '{:+.1f}%'}), use_container_width=True)

            with tab2:
                st.subheader(f"📋 2026 Strategic Plan for {selected_cust}")
                months_26 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
                cols_26 = [m.strftime('%m/%Y') for m in months_26]
                pivot_list = []
                last_act_date = df[df['Order qty.(A)'] > 0]['ds'].max()

                for p in top_prods:
                    # Lấy đúng Adj đã lưu cho từng sản phẩm p
                    p_adj = st.session_state.adj_dict.get(p, 0)
                    product_cies = cust_df[cust_df[mat_col] == p][cie_col].unique()
                    
                    for c in product_cies:
                        q_growth = get_quarterly_growth_logic(cust_df, p, cie_col, c, mat_col)
                        f_growth = q_growth + (p_adj/100)
                        
                        row = {'Product': p, 'CIE': str(c), 'Growth (Incl. Adj)': f"{f_growth*100:.1f}%"}
                        for m_date in months_26:
                            m_idx, m_str = m_date.month, m_date.strftime('%m/%Y')
                            q_idx = (m_idx - 1) // 3 + 1
                            act_26 = cust_df[(cust_df[mat_col]==p) & (cust_df[cie_col]==c) & (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2026)]['Order qty.(A)'].sum()
                            
                            if act_26 > 0:
                                row[m_str] = act_26
                            elif m_date > last_act_date:
                                avg_25 = get_actual_avg_qty(cust_df, 2025, q_idx, p, cie_col, c, mat_col)
                                row[m_str] = round(avg_25 * (1 + f_growth), 0)
                            else: row[m_str] = 0
                        pivot_list.append(row)
                
                if pivot_list:
                    res_df = pd.DataFrame(pivot_list)
                    total_row = {'Product': 'GRAND TOTAL', 'CIE': '', 'Growth (Incl. Adj)': ''}
                    for col in cols_26: total_row[col] = res_df[col].sum()
                    res_df = pd.concat([res_df, pd.DataFrame([total_row])], ignore_index=True)
                    st.dataframe(res_df.style.apply(lambda x: ['background: #e6f3ff; font-weight: bold' if x['Product']=='GRAND TOTAL' else '' for _ in x], axis=1).format("{:,.0f}", subset=cols_26), use_container_width=True)
                    st.download_button("📥 Export Plan", data=res_df.to_csv(index=False).encode('utf-8-sig'), file_name="Custom_Strategic_Plan.csv")
