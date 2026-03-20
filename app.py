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

@st.cache_data(ttl=3600)
def get_detailed_analysis(cust_df, prod_name):
    p_df = cust_df[cust_df['Material name'] == prod_name].copy()
    if len(p_df) < 5: return None, None, 0.0
    
    p_df['m'] = p_df['ds'].dt.to_period('M').dt.to_timestamp()
    df_p = p_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
    
    # Huấn luyện AI dựa trên dữ liệu QUÁ KHỨ (trước 2026) để kiểm tra độ chính xác của T1-T3 2026
    df_train = df_p[df_p['ds'] < '2026-01-01']
    if len(df_train) < 2: return None, None, 0.0

    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_train)
    future = m.make_future_dataframe(periods=12, freq='MS')
    fcst = m.predict(future)
    
    return df_p, fcst[['ds', 'yhat']], 0.0 # Trả về cả thực tế và dự báo

df = load_data()

if df is not None:
    st.sidebar.header("⚡ Settings")
    cust_col = st.sidebar.selectbox("Customer:", [c for c in df.columns if 'Customer' in c] or [df.columns[0]])
    cie_col = st.sidebar.selectbox("CIE/Color:", [c for c in df.columns if 'CIE' in c] or [df.columns[1]])
    
    cust_list = sorted(df[cust_col].unique())
    selected_cust = st.sidebar.selectbox("1. Select Customer:", ["-- Select --"] + cust_list)

    if selected_cust != "-- Select --":
        cust_df = df[df[cust_col] == selected_cust].copy()
        sales = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        top_prods = sales['Material name'].unique()[:20]
        
        tab1, tab2 = st.tabs(["📊 Digital Variance Analysis", "📋 2026 Production Plan"])

        with tab1:
            selected_prod = st.selectbox("Select Product to Verify AI Accuracy:", top_prods)
            actuals_all, fcst_res, _ = get_detailed_analysis(cust_df, selected_prod)
            
            if fcst_res is not None:
                # Lọc dữ liệu T1-T3/2026 để so sánh
                act_26 = actuals_all[actuals_all['ds'].dt.year == 2026].copy()
                fcst_26 = fcst_res[fcst_res['ds'].dt.year == 2026].copy()
                
                comparison = pd.merge(act_26, fcst_26, on='ds', how='inner')
                comparison['Variance (Qty)'] = comparison['y'] - comparison['yhat']
                comparison['Variance (%)'] = (comparison['Variance (Qty)'] / comparison['yhat']) * 100
                
                # Hiển thị bảng số hóa
                st.subheader(f"🔢 AI Performance Verification (Jan-Mar 2026)")
                
                disp_df = comparison[['ds', 'y', 'yhat', 'Variance (Qty)', 'Variance (%)']].copy()
                disp_df.columns = ['Month', 'Actual Qty', 'AI Forecasted', 'Diff (Qty)', 'Diff (%)']
                
                # Định dạng màu sắc cho bảng
                def color_variance(val):
                    color = 'red' if abs(val) > 20 else 'green'
                    return f'color: {color}'

                st.dataframe(disp_df.style.format({
                    'Month': lambda x: x.strftime('%m/%Y'),
                    'Actual Qty': '{:,.0f}',
                    'AI Forecasted': '{:,.0f}',
                    'Diff (Qty)': '{:,.0f}',
                    'Diff (%)': '{:.1f}%'
                }).applymap(color_variance, subset=['Diff (%)']), use_container_width=True)
                
                # Biểu đồ trực quan hóa sự sai lệch
                fig = go.Figure()
                fig.add_trace(go.Bar(x=disp_df['Month'], y=disp_df['Actual Qty'], name="Actual", marker_color='blue'))
                fig.add_trace(go.Bar(x=disp_df['Month'], y=disp_df['AI Forecasted'], name="AI Forecast", marker_color='orange'))
                fig.update_layout(title="Actual vs AI Forecast (Q1 2026)", barmode='group')
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.info("Kế hoạch sản xuất cho các tháng còn lại dựa trên sai số đã phân tích ở Tab 1.")
            # Logic Tab 2 giữ nguyên để tính production plan...
