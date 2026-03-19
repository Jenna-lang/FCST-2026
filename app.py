import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="AI Supply Chain Advisor", layout="wide")
st.title("🤖 AI Strategic Advisor: History & 2026 Forecast")

@st.cache_data
def load_data():
    df = pd.read_excel('AICheck.xlsx')
    df.columns = [str(col).strip() for col in df.columns]
    df['ds'] = pd.to_datetime(df['Requested deliv. date'])
    return df

try:
    df = load_data()
    
    # 2. Sidebar Configuration
    st.sidebar.header("Data Settings")
    cust_col = st.sidebar.selectbox("Customer Name Column:", df.columns)
    cie_col = st.sidebar.selectbox("CIE Color Column:", df.columns)
    
    # Per-Customer Pareto Filter
    cust_list = sorted(df[cust_col].unique().astype(str))
    selected_cust = st.selectbox("1. Select Customer:", cust_list)

    if selected_cust:
        # Filter data for this specific customer first
        cust_df = df[df[cust_col].astype(str) == selected_cust].copy()
        
        # Pareto Logic (80/20) for this customer
        sales = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        sales['Cum_Pct'] = sales['M USD'].cumsum() / sales['M USD'].sum()
        top_prods = sales[sales['Cum_Pct'] <= 0.81]['Material name'].unique()
        
        selected_prod = st.selectbox(f"2. Select Top Product for {selected_cust}:", top_prods)

        if selected_prod:
            prod_df = cust_df[cust_df['Material name'] == selected_prod].copy()
            cie_options = sorted(prod_df[cie_col].unique().astype(str))
            selected_cies = st.multiselect("3. Select CIE Color Codes:", cie_options, default=cie_options[:1])

            if selected_cies:
                fig = go.Figure()
                forecast_results = []
                ai_insights = []
                
                for cie in selected_cies:
                    cie_df = prod_df[prod_df[cie_col].astype(str) == cie].copy()
                    cie_df['m'] = cie_df['ds'].dt.to_period('M').dt.to_timestamp()
                    actual = cie_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
                    
                    if len(actual) >= 2:
                        # Prophet Model
                        m = Prophet(yearly_seasonality=True).fit(actual)
                        future = m.make_future_dataframe(periods=12, freq='MS')
                        fcst = m.predict(future)
                        fcst['yhat'] = fcst['yhat'].clip(lower=0)
                        
                        # --- Visualization ---
                        # Actual History (Solid line)
                        fig.add_trace(go.Scatter(x=actual['ds'], y=actual['y'], mode='lines+markers', name=f"Actual - {cie}"))
                        # Forecast (Dashed line)
                        f_only = fcst[fcst['ds'] > actual['ds'].max()]
                        fig.add_trace(go.Scatter(x=f_only['ds'], y=f_only['yhat'], mode='lines', 
                                                 line=dict(dash='dash'), name=f"Forecast - {cie}"))
                        
                        # --- AI Advisor Insights ---
                        avg_past = actual['y'].tail(12).mean()
                        max_fcst = f_only[f_only['ds'].dt.year == 2026]['yhat'].max()
                        growth = ((max_fcst - avg_past) / avg_past * 100) if avg_past > 0 else 0
                        status = "Increasing" if growth > 5 else "Decreasing" if growth < -5 else "Stable"
                        ai_insights.append(f"**CIE {cie}**: Demand is {status} ({growth:.1f}% vs last 12m).")
                        
                        f26 = fcst[fcst['ds'].dt.year == 2026][['ds', 'yhat']].copy()
                        f26['CIE'] = cie
                        forecast_results.append(f26)
                
                # 4. Display AI Advisor
                st.divider()
                st.subheader("💡 AI Strategic Advisor Insights")
                for insight in ai_insights:
                    st.info(insight)
                
                # 5. Display Chart & Data Table
                fig.update_layout(title=f"Sales History & 2026 Forecast for {selected_prod}", xaxis_title="Timeline", yaxis_title="Quantity (Pcs)")
                st.plotly_chart(fig, use_container_width=True)
                
                if forecast_results:
                    st.subheader("Detailed Forecast Table (2026)")
                    final_table = pd.concat(forecast_results)
                    final_table['Month'] = final_table['ds'].dt.strftime('%m/%Y')
                    display_df = final_table[['Month', 'CIE', 'yhat']].rename(columns={'yhat': 'Qty (Pcs)'})
                    # Format numbers with commas
                    st.dataframe(display_df.style.format("{:,.0f}", subset=['Qty (Pcs)']), use_container_width=True)
            else:
                st.info("Please select a CIE code.")

except Exception as e:
    st.error(f"Critical System Error: {e}")
    # --- PHẦN CŨ CỦA BẠN (Giữ nguyên) ---
df = pd.read_excel("forecast_led_lighting.xlsx") 
# Giả sử df của bạn đã có cột 'Actual' và 'Forecast'

# --- PHẦN MỚI CHÈN THÊM (Chỉ thêm, không thay thế) ---
import numpy as np
import plotly.graph_objects as go

# Tính toán (không làm thay đổi df gốc)
mask = df['Actual'] != 0
mape = np.mean(np.abs((df['Actual'][mask] - df['Forecast'][mask]) / df['Actual'][mask])) * 100

# Hiển thị thêm lên giao diện
st.divider() # Vạch ngăn cách giữa phần cũ và phần mới
st.subheader("📊 Phân tích độ chính xác")
st.metric("Chỉ số MAPE", f"{mape:.2f}%")

# Vẽ biểu đồ so sánh (Tận dụng dữ liệu cũ để vẽ)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Actual'], name='Thực tế'))
fig.add_trace(go.Scatter(x=df.index, y=df['Forecast'], name='Dự báo', line=dict(dash='dash')))
st.plotly_chart(fig)
