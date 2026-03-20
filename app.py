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
        df['ds'] = pd.to_datetime(df['Requested deliv. date'], errors='coerce')
        df = df.dropna(subset=['ds'])
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def calculate_yoy_metrics(cust_df, prod_name):
    """Calculates growth based on months with actual 2026 data compared to 2025 same period"""
    p_df = cust_df[cust_df['Material name'] == prod_name].copy()
    act_2026_full = p_df[p_df['ds'].dt.year == 2026].groupby(p_df['ds'].dt.month)['Order qty.(A)'].sum()
    active_months = act_2026_full.index.tolist()
    
    if not active_months: return 0.0, 0.0
    
    # Same period 2025
    act_2025_same = p_df[(p_df['ds'].dt.year == 2025) & (p_df['ds'].dt.month.isin(active_months))].groupby(p_df['ds'].dt.month)['Order qty.(A)'].sum()
    
    sum_26 = act_2026_full.sum()
    sum_25 = act_2025_same.sum()
    
    yoy_growth = (sum_26 - sum_25) / sum_25 if sum_25 > 0 else 0.0
    run_rate = act_2026_full.mean() # Fallback for new items
    
    return yoy_growth, run_rate

df = load_data()

if df is not None:
    st.sidebar.header("⚡ Settings")
    cust_col = st.sidebar.selectbox("Customer:", [c for c in df.columns if 'Customer' in c] or [df.columns[0]])
    cie_col = st.sidebar.selectbox("CIE/Color:", [c for c in df.columns if 'CIE' in c] or [df.columns[1]])
    adj_growth = st.sidebar.slider("Manual Adjustment (%)", -50, 50, 0)
    
    selected_cust = st.sidebar.selectbox("1. Select Customer:", ["-- Select --"] + sorted(df[cust_col].unique()))

    if selected_cust != "-- Select --":
        cust_df = df[df[cust_col] == selected_cust].copy()
        
        # --- PARETO 85% RULE ---
        rev = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        rev['Cum_Pct'] = (rev['M USD'].cumsum() / rev['M USD'].sum()) * 100
        top_prods = rev[rev['Cum_Pct'] <= 85]['Material name'].unique()[:20]
        
        st.sidebar.success(f"📈 Pareto: Tracking {len(top_prods)} Strategic Items")

        tab1, tab2 = st.tabs(["📊 Performance & Trends", "📋 2026 Production Plan"])

        with tab1:
            selected_prod = st.selectbox("Select Product to Analyze:", top_prods)
            yoy_g, r_rate = calculate_yoy_metrics(cust_df, selected_prod)
            
            p_df = cust_df[cust_df['Material name'] == selected_prod].copy()
            p_df['m'] = p_df['ds'].dt.to_period('M').dt.to_timestamp()
            df_plot = p_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
            
            # Prophet Forecast
            m = Prophet(yearly_seasonality=True).fit(df_plot)
            fcst = m.predict(m.make_future_dataframe(periods=12, freq='MS'))
            
            # Trend Chart with Growth Indicator
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_plot[df_plot['ds'].dt.year<2026]['ds'], y=df_plot['y'], name="History", line=dict(color='gray')))
            fig.add_trace(go.Scatter(x=df_plot[df_plot['ds'].dt.year==2026]['ds'], y=df_plot['y'], name="Actual 2026", mode='markers+lines', line=dict(color='blue', width=3)))
            fig.add_trace(go.Scatter(x=fcst[fcst['ds'].dt.year==2026]['ds'], y=fcst['yhat'], name="AI Forecast", line=dict(dash='dash', color='orange')))
            
            fig.update_layout(title=f"Trend: {selected_prod} (YoY Growth: {yoy_g*100:.1f}%)", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # --- DIGITAL VARIANCE WITH RED ALERT (>20%) ---
            st.subheader("🔢 Q1 Digital Variance Analysis")
            q1_act = df_plot[df_plot['ds'].dt.year == 2026]
            q1_fcst = fcst[fcst['ds'].dt.year == 2026]
            comp = pd.merge(q1_act, q1_fcst, on='ds', how='inner')
            
            if not comp.empty:
                comp['Variance %'] = ((comp['y'] - comp['yhat']) / comp['yhat']) * 100
                
                # Apply Red formatting for >20% deviation
                def color_variance(val):
                    color = 'red' if abs(val) > 20 else 'green'
                    return f'color: {color}; font-weight: bold'

                st.dataframe(comp[['ds','y','yhat','Variance %']].style.format({
                    'y': '{:,.0f}', 'yhat': '{:,.0f}', 'Variance %': '{:.1f}%'
                }).applymap(color_variance, subset=['Variance %']), use_container_width=True)

        with tab2:
            st.subheader("📋 2026 Plan: YoY Strategic Logic")
            months_26 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
            cols_26 = [m.strftime('%m/%Y') for m in months_26]
            
            pivot_list = []
            for p in top_prods:
                y_growth, r_rate_fallback = calculate_yoy_metrics(cust_df, p)
                final_factor = 1 + y_growth + (adj_growth / 100)
                
                cies = cust_df[cust_df['Material name'] == p][cie_col].unique()
                for c in cies:
                    row = {'Product': p, 'CIE': c, 'YoY Growth': f"{y_growth*100:.1f}%"}
                    for m_date in months_26:
                        m_idx, m_str = m_date.month, m_date.strftime('%m/%Y')
                        
                        # 1. Actuals 2026
                        act = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2026)]['Order qty.(A)'].sum()
                        
                        if act > 0:
                            row[m_str] = act
                        elif m_idx > 3: # Forecast
                            # 2. YoY Logic (Same month 2025)
                            h25 = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2025)]['Order qty.(A)'].sum()
                            if h25 > 0:
                                row[m_str] = round(h25 * final_factor, 0)
                            else:
                                # 3. New Item Logic (Run-rate 2026 fallback)
                                row[m_str] = round(r_rate_fallback * final_factor, 0)
                        else: row[m_str] = 0
                    pivot_list.append(row)
            
            res_df = pd.DataFrame(pivot_list)
            # Grand Total
            total_row = {'Product': 'GRAND TOTAL', 'CIE': '---', 'YoY Growth': '---'}
            for col in cols_26: total_row[col] = res_df[col].sum()
            res_df = pd.concat([res_df, pd.DataFrame([total_row])], ignore_index=True)
            
            st.dataframe(res_df.style.apply(lambda x: ['font-weight:bold; background:#e6f3ff' if x['Product']=='GRAND TOTAL' else '' for _ in x], axis=1).format("{:,.0f}", subset=cols_26), use_container_width=True)
