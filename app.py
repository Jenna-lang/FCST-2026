import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. System Configuration
st.set_page_config(page_title="AI Supply Chain Advisor 2026", layout="wide")

@st.cache_data(ttl=3600)
def load_data():
    try:
        # Loading the specific Excel file
        df = pd.read_excel('AICheck.xlsx')
        df.columns = [str(col).strip() for col in df.columns]
        # Clean column names and string data to prevent matching errors
        for col in df.select_dtypes(['object']).columns:
            df[col] = df[col].astype(str).str.strip()
            
        # Standardize date format
        df['ds'] = pd.to_datetime(df['Requested deliv. date'], errors='coerce')
        df = df.dropna(subset=['ds'])
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# AI Forecast Function - Returns detailed forecast and growth
@st.cache_data(ttl=3600)
def get_detailed_fcst(cust_df, prod_name):
    p_df = cust_df[cust_df['Material name'] == prod_name].copy()
    if len(p_df) < 5: return None, 0.0
    
    p_df['m'] = p_df['ds'].dt.to_period('M').dt.to_timestamp()
    df_p = p_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
    
    # Standardizing for Prophet
    df_prophet = df_p[['ds', 'y']].copy()
    if len(df_prophet) < 2: return None, 0.0

    try:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(df_prophet)
        future = m.make_future_dataframe(periods=12, freq='MS')
        fcst = m.predict(future)
        
        # Calculate AI Growth: 2026 (Actual + Forecast) vs 2025 Total
        total_25 = df_p[df_p['ds'].dt.year == 2025]['y'].sum()
        act_26 = df_p[df_p['ds'].dt.year == 2026]['y'].sum()
        
        last_date = df_p['ds'].max()
        fcst_26 = fcst[(fcst['ds'].dt.year == 2026) & (fcst['ds'] > last_date)]['yhat'].sum()
        
        total_26_mixed = act_26 + fcst_26
        ai_growth = ((total_26_mixed - total_25) / total_25) if total_25 > 0 else 0.0
        
        return fcst[['ds', 'yhat']], ai_growth
    except:
        return None, 0.0

df = load_data()

if df is not None:
    # 2. Sidebar Configuration
    st.sidebar.header("⚡ Settings")
    cust_col = st.sidebar.selectbox("Customer Column:", [c for c in df.columns if 'Customer' in c] or [df.columns[0]])
    cie_col = st.sidebar.selectbox("CIE/Color Column:", [c for c in df.columns if 'CIE' in c] or [df.columns[1]])
    
    # Manual Adjustment Slider on top of AI
    adj_growth = st.sidebar.slider("AI Adjustment (%)", -50, 50, 0, help="Add/Subtract from AI's growth prediction")
    
    cust_list = sorted(df[cust_col].unique())
    selected_cust = st.sidebar.selectbox("1. Select Customer:", ["-- Select --"] + cust_list)

    if selected_cust != "-- Select --":
        cust_df = df[df[cust_col] == selected_cust].copy()
        
        # Pareto 80/20 Analysis (Top Products)
        sales = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        sales['Cum_Pct'] = sales['M USD'].cumsum() / (sales['M USD'].sum() or 1)
        top_prods = sales[sales['Cum_Pct'] <= 0.85]['Material name'].unique()
        
        # Pre-calculate AI Growth for the table in Tab 2 to avoid lags
        with st.spinner("Analyzing AI trends for Pareto products..."):
            growth_map = {p: get_detailed_fcst(cust_df, p)[1] for p in top_prods}
            # Remove products where AI failed (returned 0 growth but len<5)
            top_prods = [p for p in top_prods if p in growth_map and growth_map[p] is not None]

        tab1, tab2 = st.tabs(["📊 Individual Analysis", "📋 2026 Production Plan"])

        # --- TAB 1: INDIVIDUAL ANALYSIS WITH CHARTS ---
        with tab1:
            selected_prod = st.selectbox("Select Product to Analyze:", top_prods)
            
            # Feature: NEW Annual Target Slider for this product
            target_annual = st.sidebar.number_input(f"Set Annual Target for {selected_prod} (Pcs):", value=100000, step=1000)
            target_monthly = target_annual / 12 if target_annual > 0 else 0
            
            fcst_res, ai_g = get_detailed_fcst(cust_df, selected_prod)
            
            if fcst_res is not None:
                final_g = ai_g + (adj_growth / 100)
                st.metric(f"Projected Growth ({selected_prod})", f"{final_g*100:.1f}%", delta=f"{adj_growth}% Adj")
                
                # Plotly Chart
                fig = go.Figure()
                
                # Pre-processing actuals
                p_df = cust_df[cust_df['Material name'] == selected_prod].copy()
                p_df['m'] = p_df['ds'].dt.to_period('M').dt.to_timestamp()
                actuals = p_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
                
                # 1. Historical Actuals (Gray Line)
                history = actuals[actuals['ds'].dt.year < 2026]
                if not history.empty:
                    fig.add_trace(go.Scatter(x=history['ds'], y=history['y'], name="History (2023-25)", line=dict(color='gray', width=1, dash='dot')))
                
                # 2. 2026 Actuals (Q1) (Blue Line)
                act_26 = actuals[actuals['ds'].dt.year == 2026]
                if not act_26.empty:
                    fig.add_trace(go.Scatter(x=act_26['ds'], y=act_26['y'], name="Actual 2026 (Q1)", mode='lines+markers', line=dict(color='blue', width=3)))
                
                # 3. AI Forecast (Orange Dashed)
                last_act_date = actuals['ds'].max()
                future_fcst = fcst_res[fcst_res['ds'] >= '2026-01-01']
                if not future_fcst.empty:
                    # For visualization, we keep forecast from T1 but mark it clearly
                    fig.add_trace(go.Scatter(x=future_fcst['ds'], y=future_fcst['yhat'], name="AI Forecast 2026", line=dict(color='orange', dash='dash')))
                
                # 4. Target Line (Green Dashed)
                months_26_dates = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
                target_data = [target_monthly] * 12
                fig.add_trace(go.Scatter(x=months_26_dates, y=target_data, name="Annual Target Line", line=dict(color='green', dash='dot')))
                
                fig.update_layout(title=f"Actual vs Forecast vs Target: {selected_prod}", hovermode="x unified", xaxis_title="Timeline", yaxis_title="Quantity (Pcs)")
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning(f"No forecast available for {selected_prod} due to insufficient historical data.")

        # --- TAB 2: ROBUST PRODUCTION PLAN ---
        with tab2:
            st.subheader("📋 2026 Supply Plan (AI Growth per Item + Adjustment)")
            
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
                # Combine AI Growth and Manual Adjustment
                ai_growth_val = growth_map.get(p, 0.0)
                final_factor = 1 + ai_growth_val + (adj_growth / 100)
                
                row = {'Product': p, 'CIE': c, 'Applied Growth': f"{(final_factor-1)*100:.1f}%"}
                
                for m_date in months_26:
                    m_idx, m_str = m_date.month, m_date.strftime('%m/%Y')
                    if (p, c, m_idx) in act_26_map:
                        row[m_str] = act_26_map[(p, c, m_idx)]
                    elif m_idx > 3:
                        if (p, c, m_idx) in act_25_map:
                            row[m_str] = round(act_25_map[(p, c, m_idx)] * final_factor, 0)
                        elif (p, c) in run_rate_map:
                            row[m_str] = round(run_rate_map[(p, c)] * final_factor, 0)
                        else: row[m_str] = 0
                    else: row[m_str] = 0
                pivot_list.append(row)
            
            if pivot_list:
                res_df = pd.DataFrame(pivot_list)
                
                # Add GRAND TOTAL row
                total_row = {'Product': 'GRAND TOTAL', 'CIE': '---
