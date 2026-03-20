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

@st.cache_resource
def fast_forecast(data_series):
    if len(data_series) < 2: return None
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(data_series)
    future = m.make_future_dataframe(periods=12, freq='MS')
    fcst = m.predict(future)
    return fcst[['ds', 'yhat']]

df = load_data()

if df is not None:
    # 2. Sidebar Configuration
    st.sidebar.header("⚡ Settings")
    cust_col = st.sidebar.selectbox("Customer Column:", [c for c in df.columns if 'Customer' in c] or [df.columns[0]])
    cie_col = st.sidebar.selectbox("CIE/Color Column:", [c for c in df.columns if 'CIE' in c] or [df.columns[1]])
    
    # NEW FEATURE: Manual Growth Rate Slider
    user_growth = st.sidebar.slider("Target Growth Rate (%)", min_value=-50, max_value=100, value=20)
    g_factor = 1 + (user_growth / 100)
    
    cust_list = sorted(df[cust_col].unique())
    selected_cust = st.sidebar.selectbox("1. Select Customer:", ["-- Select --"] + cust_list)

    if selected_cust != "-- Select --":
        cust_df = df[df[cust_col] == selected_cust].copy()
        
        # Pareto 80/20 Analysis
        sales = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        sales['Cum_Pct'] = sales['M USD'].cumsum() / (sales['M USD'].sum() or 1)
        top_prods = sales[sales['Cum_Pct'] <= 0.85]['Material name'].unique()
        
        tab1, tab2 = st.tabs(["📊 Trend Preview", "📋 2026 Production Plan"])

        # --- TAB 1: PREVIEW ---
        with tab1:
            selected_prod = st.selectbox("Select Product to Preview:", top_prods)
            prod_df = cust_df[cust_df['Material name'] == selected_prod].copy()
            prod_df['m'] = prod_df['ds'].dt.to_period('M').dt.to_timestamp()
            actual_all = prod_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
            
            res = fast_forecast(actual_all)
            if res is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=actual_all['ds'], y=actual_all['y'], name="Actual", line=dict(color='#1f77b4')))
                fig.add_trace(go.Scatter(x=res[res['ds'] > actual_all['ds'].max()]['ds'], y=res[res['ds'] > actual_all['ds'].max()]['yhat'], 
                                         line=dict(dash='dash', color='#ff7f0e'), name="AI Prediction"))
                st.plotly_chart(fig, use_container_width=True)

        # --- TAB 2: PRODUCTION PLAN WITH TOTAL ROW ---
        with tab2:
            st.subheader(f"📋 2026 Plan with {user_growth}% Growth Target")
            
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
                
                for m_date in months_26:
                    m_idx = m_date.month
                    m_str = m_date.strftime('%m/%Y')
                    
                    if (p, c, m_idx) in act_26_map:
                        row[m_str] = act_26_map[(p, c, m_idx)]
                    elif m_idx > 3:
                        if (p, c, m_idx) in act_25_map:
                            row[m_str] = round(act_25_map[(p, c, m_idx)] * g_factor, 0)
                        elif (p, c) in run_rate_map:
                            row[m_str] = round(run_rate_map[(p, c)] * g_factor, 0)
                        else:
                            row[m_str] = 0
                    else:
                        row[m_str] = 0
                pivot_list.append(row)
            
            if pivot_list:
                res_df = pd.DataFrame(pivot_list)
                
                # NEW FEATURE: Add Total Row
                total_row = {'Product': 'TOTAL', 'CIE': '---'}
                for col in cols_26:
                    total_row[col] = res_df[col].sum()
                
                res_df = pd.concat([res_df, pd.DataFrame([total_row])], ignore_index=True)
                
                # Apply styling to make Total row bold
                def bold_total(row):
                    return ['font-weight: bold; background-color: #f0f2f6' if row['Product'] == 'TOTAL' else '' for _ in row]

                st.dataframe(res_df.style.apply(bold_total, axis=1).format("{:,.0f}", subset=cols_26), use_container_width=True)
                st.download_button("📥 Export Plan (CSV)", res_df.to_csv(index=False).encode('utf-8-sig'), "Supply_Plan_2026.csv")
