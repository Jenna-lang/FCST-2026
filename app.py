import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. System Setup
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

def calculate_yoy_growth(cust_df, prod_name):
    p_df = cust_df[cust_df['Material name'] == prod_name].copy()
    act_2026 = p_df[p_df['ds'].dt.year == 2026].groupby(p_df['ds'].dt.month)['Order qty.(A)'].sum()
    active_months = act_2026.index.tolist()
    if not active_months: return 0.0, 0.0
    act_2025_same = p_df[(p_df['ds'].dt.year == 2025) & (p_df['ds'].dt.month.isin(active_months))].groupby(p_df['ds'].dt.month)['Order qty.(A)'].sum()
    yoy_growth = (act_2026.sum() - act_2025_same.sum()) / act_2025_same.sum() if act_2025_same.sum() > 0 else 0.0
    return yoy_growth, act_2026.mean()

df = load_data()

if df is not None:
    st.sidebar.header("⚡ Settings")
    cust_col = st.sidebar.selectbox("Customer:", [c for c in df.columns if 'Customer' in c] or [df.columns[0]])
    cie_col = st.sidebar.selectbox("CIE/Color:", [c for c in df.columns if 'CIE' in c] or [df.columns[1]])
    adj_growth = st.sidebar.slider("Manual Adjustment (%)", -50, 50, 0)
    selected_cust = st.sidebar.selectbox("1. Select Customer:", ["-- Select --"] + sorted(df[cust_col].unique()))

    if selected_cust != "-- Select --":
        cust_df = df[df[cust_col] == selected_cust].copy()
        rev = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        rev['Cum_Pct'] = (rev['M USD'].cumsum() / rev['M USD'].sum()) * 100
        top_prods = rev[rev['Cum_Pct'] <= 85]['Material name'].unique()[:20]

        tab1, tab2 = st.tabs(["📊 Performance & AI Audit", "📋 2026 Strategic Plan"])

        with tab1:
            selected_prod = st.selectbox("Product Audit:", top_prods)
            yoy_g, r_rate = calculate_yoy_growth(cust_df, selected_prod)
            
            p_df = cust_df[cust_df['Material name'] == selected_prod].copy()
            p_df['m'] = p_df['ds'].dt.to_period('M').dt.to_timestamp()
            df_plot = p_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
            
            m = Prophet(yearly_seasonality=True).fit(df_plot)
            future = m.make_future_dataframe(periods=12, freq='MS')
            fcst = m.predict(future)
            
            # CHART
            fig = go.Figure()
            act_2026 = df_plot[df_plot['ds'].dt.year == 2026]
            fig.add_trace(go.Scatter(x=act_2026['ds'], y=act_2026['y'], name="Actual 2026", mode='lines+markers', line=dict(color='blue', width=4)))
            fcst_2026 = fcst[fcst['ds'].dt.year == 2026]
            fig.add_trace(go.Scatter(x=fcst_2026['ds'], y=fcst_2026['yhat'], name="AI Trend Forecast", line=dict(dash='dash', color='orange', width=2)))
            fig.add_trace(go.Scatter(x=df_plot[df_plot['ds'].dt.year < 2026]['ds'], y=df_plot['y'], name="History", line=dict(color='lightgray')))
            fig.update_layout(title=f"Accuracy Audit: {selected_prod}", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # --- VARIANCE TABLE WITH YTD LOGIC ---
            st.subheader("🔢 Digital Variance Analysis (Monthly vs Lũy kế)")
            comp = pd.merge(act_2026, fcst[['ds', 'yhat']], on='ds')
            if not comp.empty:
                # Monthly Variance
                comp['Var %'] = ((comp['y'] - comp['yhat']) / comp['yhat']) * 100
                # YTD Variance (Lũy kế)
                comp['Actual_Cum'] = comp['y'].cumsum()
                comp['Forecast_Cum'] = comp['yhat'].cumsum()
                comp['YTD Var %'] = ((comp['Actual_Cum'] - comp['Forecast_Cum']) / comp['Forecast_Cum']) * 100
                
                def style_variance(val):
                    return 'color: red; font-weight: bold' if abs(val) > 20 else 'color: green'

                st.dataframe(comp[['ds','y','yhat','Var %', 'YTD Var %']].style.format({
                    'ds': lambda x: x.strftime('%m/%Y'), 'y': '{:,.0f}', 'yhat': '{:,.0f}', 
                    'Var %': '{:.1f}%', 'YTD Var %': '{:.1f}%'
                }).applymap(style_variance, subset=['Var %', 'YTD Var %']), use_container_width=True)
                
                st.info("💡 **Mẹo quyết định:** Nếu 'Var %' nhảy thất thường nhưng 'YTD Var %' ổn định (<10%), bạn không cần điều chỉnh Forecast. Chỉ can thiệp khi cả hai đều báo đỏ.")

        with tab2:
            # (Giữ nguyên logic Tab 2 như cũ)
            st.subheader("📋 2026 Strategic Plan")
            # ... [Phần code Tab 2]
