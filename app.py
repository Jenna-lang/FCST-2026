import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. System Configuration
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

# AI Forecast Function - Now returns the growth percentage
@st.cache_resource
def get_ai_growth(cust_df, prod_name):
    prod_df = cust_df[cust_df['Material name'] == prod_name].copy()
    if len(prod_df) < 2: return 0.0
    
    prod_df['m'] = prod_df['ds'].dt.to_period('M').dt.to_timestamp()
    df_prophet = prod_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
    
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=12, freq='MS')
    fcst = m.predict(future)
    
    # Calculate AI Growth: 2026 (Actual + Forecast) vs 2025 Total
    total_25 = df_prophet[df_prophet['ds'].dt.year == 2025]['y'].sum()
    act_26 = df_prophet[df_prophet['ds'].dt.year == 2026]['y'].sum()
    
    last_date = df_prophet['ds'].max()
    fcst_26 = fcst[(fcst['ds'].dt.year == 2026) & (fcst['ds'] > last_date)]['yhat'].sum()
    
    total_26 = act_26 + fcst_26
    return ((total_26 - total_25) / total_25) if total_25 > 0 else 0.0

df = load_data()

if df is not None:
    st.sidebar.header("⚡ Settings")
    cust_col = st.sidebar.selectbox("Customer Column:", [c for c in df.columns if 'Customer' in c] or [df.columns[0]])
    cie_col = st.sidebar.selectbox("CIE/Color Column:", [c for c in df.columns if 'CIE' in c] or [df.columns[1]])
    
    # FEATURE: Manual Adjustment on top of AI
    adj_growth = st.sidebar.slider("AI Adjustment (%)", -50, 50, 0, help="Add or subtract from AI's calculated growth")
    
    cust_list = sorted(df[cust_col].unique())
    selected_cust = st.sidebar.selectbox("1. Select Customer:", ["-- Select --"] + cust_list)

    if selected_cust != "-- Select --":
        cust_df = df[df[cust_col] == selected_cust].copy()
        
        # Pareto 80/20
        sales = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        sales['Cum_Pct'] = sales['M USD'].cumsum() / (sales['M USD'].sum() or 1)
        top_prods = sales[sales['Cum_Pct'] <= 0.85]['Material name'].unique()
        
        # Pre-calculate AI Growth for all Top Products
        with st.spinner('AI is analyzing trends for all products...'):
            ai_growth_map = {p: get_ai_growth(cust_df, p) for p in top_prods}

        tab1, tab2 = st.tabs(["📊 Individual Analysis", "📋 2026 Production Plan"])

        with tab1:
            selected_prod = st.selectbox("Select Product to Preview:", top_prods)
            # Display AI Growth vs Final Growth
            ai_g = ai_growth_map.get(selected_prod, 0.0)
            final_g = ai_g + (adj_growth / 100)
            st.metric("AI Predicted Growth", f"{ai_g*100:.1f}%", delta=f"{adj_growth}% Adj")
            st.write(f"**Final Applied Growth:** {final_g*100:.1f}%")

        with tab2:
            st.subheader("📋 2026 Plan (Based on AI Growth per Product)")
            
            # Data Mapping
            df_25 = cust_df[cust_df['ds'].dt.year == 2025]
            df_26 = cust_df[cust_df['ds'].dt.year == 2026]
            act_25_map = df_25.groupby(['Material name', cie_col, df_25['ds'].dt.month])['Order qty.(A)'].sum().to_dict()
            act_26_map = df_26.groupby(['Material name', cie_col, df_26['ds'].dt.month])['Order qty.(A)'].sum().to_dict()
            run_rate_map = df_26[df_26['ds'].dt.month <= 3].groupby(['Material name', cie_col])['Order qty.(A)'].mean().to_dict()

            months_26 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
            cols_26 = [m.strftime('%m/%Y') for m in months_26]
            
            all_items = cust_df[cust_df['Material name'].isin(top_prods)][['Material name', cie_col]].drop_duplicates()
            pivot_list = []
            
            for _, r in all_items.iterrows():
                p, c = r['Material name'], r[cie_col]
                row = {'Product': p, 'CIE': c}
                
                # Get item-specific growth from AI + manual adjustment
                item_g_factor = 1 + ai_growth_map.get(p, 0.0) + (adj_growth / 100)
                
                for m_date in months_26:
                    m_idx = m_date.month
                    m_str = m_date.strftime('%m/%Y')
                    
                    if (p, c, m_idx) in act_26_map:
                        row[m_str] = act_26_map[(p, c, m_idx)]
                    elif m_idx > 3:
                        if (p, c, m_idx) in act_25_map:
                            row[m_str] = round(act_25_map[(p, c, m_idx)] * item_g_factor, 0)
                        elif (p, c) in run_rate_map:
                            row[m_str] = round(run_rate_map[(p, c)] * item_g_factor, 0)
                        else: row[m_str] = 0
                    else: row[m_str] = 0
                pivot_list.append(row)
            
            if pivot_list:
                res_df = pd.DataFrame(pivot_list)
                # Total Row
                total_row = {'Product': 'GRAND TOTAL', 'CIE': '---'}
                for col in cols_26: total_row[col] = res_df[col].sum()
                res_df = pd.concat([res_df, pd.DataFrame([total_row])], ignore_index=True)
                
                st.dataframe(res_df.style.apply(lambda x: ['font-weight:bold; background:#e6f3ff' if x['Product']=='GRAND TOTAL' else '' for _ in x], axis=1).format("{:,.0f}", subset=cols_26), use_container_width=True)
