import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. System Configuration
st.set_page_config(page_title="AI Supply Chain Advisor 2026", layout="wide")

# Persistent memory for manual adjustments per product
if 'adj_dict' not in st.session_state:
    st.session_state.adj_dict = {}

# --- HELPER FUNCTIONS ---
def find_col(df, keywords):
    for k in keywords:
        for col in df.columns:
            if k.lower() in str(col).lower():
                return col
    return df.columns[0]

def get_actual_avg_qty(df, year, quarter, prod, cie_col, cie_val, mat_col):
    temp = df[(df[mat_col] == prod) & 
              (df[cie_col].astype(str) == str(cie_val)) & 
              (df['ds'].dt.year == year) & 
              (df['ds'].dt.quarter == quarter)].copy()
    monthly_sum = temp.groupby(temp['ds'].dt.month)['Order qty.(A)'].sum()
    actual_months = monthly_sum[monthly_sum > 0]
    return actual_months.mean() if not actual_months.empty else 0.0

def get_quarterly_growth_logic(cust_df, prod, cie_col, cie_val, mat_col):
    df_26 = cust_df[cust_df['ds'].dt.year == 2026].copy()
    valid_26 = df_26[df_26['Order qty.(A)'] > 0]
    if valid_26.empty: return 0.0
    latest_q_26 = valid_26['ds'].dt.quarter.max()
    avg_26 = get_actual_avg_qty(cust_df, 2026, latest_q_26, prod, cie_col, cie_val, mat_col)
    avg_25 = get_actual_avg_qty(cust_df, 2025, latest_q_26, prod, cie_col, cie_val, mat_col)
    return min(max((avg_26 / avg_25) - 1, -0.5), 0.5) if avg_25 > 0 else 0.0

# --- DATA PROCESSING ---
def process_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = [str(col).strip() for col in df.columns]
        date_c = find_col(df, ['date', 'requested'])
        qty_c = find_col(df, ['qty', 'order'])
        df['ds'] = pd.to_datetime(df[date_c], errors='coerce')
        df['Order qty.(A)'] = pd.to_numeric(df[qty_c], errors='coerce').fillna(0)
        return df.dropna(subset=['ds'])
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# --- MAIN UI ---
st.sidebar.header("📁 Supply Chain Data")
uploaded_file = st.sidebar.file_uploader("Upload AICheck.xlsx", type=['xlsx'])

if uploaded_file:
    df = process_data(uploaded_file)
    if df is not None:
        all_cols = df.columns.tolist()
        cust_col = st.sidebar.selectbox("Customer Column:", all_cols, index=all_cols.index(find_col(df, ['customer'])))
        cie_col = st.sidebar.selectbox("CIE / Item Code Column:", all_cols, index=all_cols.index(find_col(df, ['cie', 'item', 'code'])))
        mat_col = find_col(df, ['material', 'product'])
        
        selected_cust = st.sidebar.selectbox("Select Target Customer:", ["-- Select --"] + sorted(df[cust_col].unique().tolist()))

        if selected_cust != "-- Select --":
            cust_df = df[df[cust_col] == selected_cust].copy()
            rev_c = find_col(df, ['m usd', 'revenue', 'value'])
            rev = cust_df.groupby(mat_col)[rev_c].sum().sort_values(ascending=False).reset_index()
            top_prods = rev[rev[rev_c].cumsum() / rev[rev_c].sum() <= 0.86][mat_col].unique()[:50]

            tab1, tab2 = st.tabs(["📊 Performance Audit", "📋 2026 Strategic Plan"])

            with tab1:
                st.subheader("🔍 Individual Product Tuning")
                selected_prod = st.selectbox("Select Product to Audit:", top_prods)
                
                # Per-product adjustment
                curr_adj = st.session_state.adj_dict.get(selected_prod, 0)
                new_adj = st.number_input(f"Manual Adjustment for {selected_prod} (%)", value=curr_adj)
                st.session_state.adj_dict[selected_prod] = new_adj
                
                p_data = cust_df[cust_df[mat_col] == selected_prod].groupby(cust_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
                p_data['ds'] = p_data['ds'].dt.to_timestamp()
                act_26 = p_data[p_data['ds'].dt.year == 2026].rename(columns={'Order qty.(A)': 'Actual'})
                
                if len(p_data) > 2:
                    # Quick forecast for variance table
                    m = Prophet().fit(p_data.rename(columns={'ds': 'ds', 'Order qty.(A)': 'y'}))
                    fcst = m.predict(m.make_future_dataframe(periods=12, freq='MS'))
                    fcst_26 = fcst[fcst['ds'].dt.year == 2026][['ds', 'yhat']].rename(columns={'yhat': 'AI Forecast'})
                    
                    v_df = pd.merge(act_26, fcst_26, on='ds', how='inner')
                    if not v_df.empty:
                        v_df['Variance %'] = ((v_df['Actual'] - v_df['AI Forecast']) / v_df['AI Forecast']) * 100
                        
                        # Add AVERAGE row
                        avg_row = pd.DataFrame({
                            'ds': ["AVERAGE"],
                            'Actual': [v_df['Actual'].mean()], 
                            'AI Forecast': [v_df['AI Forecast'].mean()], 
                            'Variance %': [v_df['Variance %'].mean()]
                        })
                        
                        v_display = pd.concat([v_df, avg_row], ignore_index=True)
                        
                        st.subheader("🔢 Performance Variance Analysis")
                        st.dataframe(v_display.style.format({
                            'ds': lambda x: x.strftime('%m/%Y') if hasattr(x, 'strftime') else x,
                            'Actual': '{:,.0f}',
                            'AI Forecast': '{:,.0f}',
                            'Variance %': '{:+.1f}%'
                        }).apply(lambda x: ['background: #f0f2f6; font-weight: bold'] * len(x) if x['ds'] == "AVERAGE" else [''] * len(x), axis=1), use_container_width=True)

            with tab2:
                st.subheader(f"📋 2026 Strategic Plan: {selected_cust}")
                months_26 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
                cols_str = [m.strftime('%m/%Y') for m in months_26]
                pivot_data = []
                last_date = df[df['Order qty.(A)'] > 0]['ds'].max()

                for p in top_prods:
                    p_adj = st.session_state.adj_dict.get(p, 0)
                    p_cies = cust_df[cust_df[mat_col] == p][cie_col].unique()
                    for c in p_cies:
                        q_grow = get_quarterly_growth_logic(cust_df, p, cie_col, c, mat_col)
                        total_grow = q_grow + (p_adj/100)
                        row = {'Product': p, 'CIE': str(c), 'Growth Rate': f"{total_grow*100:.1f}%"}
                        for m_dt in months_26:
                            m_s = m_dt.strftime('%m/%Y')
                            actual = cust_df[(cust_df[mat_col]==p) & (cust_df[cie_col].astype(str)==str(c)) & 
                                            (cust_df['ds'].dt.month==m_dt.month) & (cust_df['ds'].dt.year==2026)]['Order qty.(A)'].sum()
                            if actual > 0: row[m_s] = actual
                            elif m_dt > last_date:
                                avg_25 = get_actual_avg_qty(cust_df, 2025, (m_dt.month-1)//3+1, p, cie_col, c, mat_col)
                                row[m_s] = round(avg_25 * (1 + total_grow), 0)
                            else: row[m_s] = 0
                        pivot_data.append(row)
                
                if pivot_data:
                    res_df = pd.DataFrame(pivot_data)
                    total_row = {'Product': 'GRAND TOTAL', 'CIE': '', 'Growth Rate': ''}
                    for c_s in cols_str: total_row[c_s] = res_df[c_s].sum()
                    res_df = pd.concat([res_df, pd.DataFrame([total_row])], ignore_index=True)
                    st.dataframe(res_df.style.apply(lambda x: ['background: #e6f3ff; font-weight: bold' if x['Product']=='GRAND TOTAL' else '' for _ in x], axis=1).format("{:,.0f}", subset=cols_str), use_container_width=True)
                    st.download_button("📥 Export CSV", res_df.to_csv(index=False).encode('utf-8-sig'), "Supply_Chain_Plan_2026.csv")
