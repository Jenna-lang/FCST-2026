import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. Cấu hình hệ thống
st.set_page_config(page_title="AI Supply Chain Advisor 2026", layout="wide")

# --- HÀM XỬ LÝ DỮ LIỆU ---
def process_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = [str(col).strip() for col in df.columns]
        date_col = 'Requested deliv. date'
        qty_col = 'Order qty.(A)'
        
        if date_col in df.columns and qty_col in df.columns:
            df['ds'] = pd.to_datetime(df[date_col], errors='coerce')
            df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
            df = df.dropna(subset=['ds'])
            return df
        else:
            st.error(f"Thiếu cột '{date_col}' hoặc '{qty_col}' trong file!")
            return None
    except Exception as e:
        st.error(f"Lỗi định dạng file: {e}")
        return None

def calculate_full_metrics(cust_df, prod_name):
    p_df = cust_df[cust_df['Material name'] == prod_name].copy()
    if p_df.empty: return 0.0, 0.0, 0.0
    
    monthly_all = p_df.groupby(p_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
    monthly_all['ds_ts'] = monthly_all['ds'].dt.to_timestamp()
    
    # 1. Avg Growth
    monthly_all['Pct_Change'] = monthly_all['Order qty.(A)'].pct_change() * 100
    avg_growth = monthly_all['Pct_Change'].mean()
    
    # 2. Metrics 2026
    df_2026 = monthly_all[monthly_all['ds_ts'].dt.year == 2026]
    act_2026_sum = df_2026['Order qty.(A)'].sum()
    run_rate_26 = df_2026['Order qty.(A)'].mean() if not df_2026.empty else 0.0
    
    # 3. YoY Growth
    months_active_26 = df_2026['ds_ts'].dt.month.tolist()
    act_2025_same = monthly_all[(monthly_all['ds_ts'].dt.year == 2025) & 
                                (monthly_all['ds_ts'].dt.month.isin(months_active_26))]['Order qty.(A)'].sum()
    yoy_growth = (act_2026_sum - act_2025_same) / act_2025_same if act_2025_same > 0 else 0.0
    
    return avg_growth, yoy_growth, run_rate_26

# --- SIDEBAR ---
st.sidebar.header("📁 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload AICheck.xlsx", type=['xlsx'])

if uploaded_file:
    df = process_data(uploaded_file)
    if df is not None:
        cust_col = st.sidebar.selectbox("Customer:", [c for c in df.columns if 'Customer' in c] or [df.columns[0]])
        cie_col = st.sidebar.selectbox("CIE/Color:", [c for c in df.columns if 'CIE' in c] or [df.columns[1]])
        adj_growth = st.sidebar.slider("Manual Adjustment (%)", -50, 50, 0)
        selected_cust = st.sidebar.selectbox("Select Customer:", ["-- Select --"] + sorted(df[cust_col].unique()))

        if selected_cust != "-- Select --":
            cust_df = df[df[cust_col] == selected_cust].copy()
            
            # Pareto 85%
            rev = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
            rev['Cum_Pct'] = rev['M USD'].cumsum() / rev['M USD'].sum()
            top_prods = rev[rev['Cum_Pct'] <= 0.86]['Material name'].unique()[:20]

            tab1, tab2 = st.tabs(["📊 Performance & AI Audit", "📋 2026 Strategic Plan"])

            with tab1:
                selected_prod = st.selectbox("Product Audit:", top_prods)
                avg_g, yoy_g, r_rate = calculate_full_metrics(cust_df, selected_prod)
                
                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Avg Growth (History)", f"{avg_g:.1f}%")
                m2.metric("YoY Growth (26 vs 25)", f"{yoy_g*100:.1f}%")
                m3.metric("Run-rate 2026", f"{r_rate:,.0f}")

                # BIỂU ĐỒ (2023 -> 2026)
                p_df_plot = cust_df[cust_df['Material name'] == selected_prod].copy()
                df_monthly = p_df_plot.groupby(p_df_plot['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
                df_monthly['ds'] = df_monthly['ds'].dt.to_timestamp()
                df_monthly = df_monthly.rename(columns={'Order qty.(A)': 'y'})

                m = Prophet(yearly_seasonality=True).fit(df_monthly)
                future = m.make_future_dataframe(periods=12, freq='MS')
                fcst = m.predict(future)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_monthly['ds'], y=df_monthly['y'], name="Actual (History + 2026)", line=dict(color='blue', width=3)))
                fcst_26 = fcst[fcst['ds'].dt.year == 2026]
                fig.add_trace(go.Scatter(x=fcst_26['ds'], y=fcst_26['yhat'], name="AI Forecast 2026", line=dict(dash='dash', color='orange')))
                st.plotly_chart(fig, use_container_width=True)

                # --- MỤC SO SÁNH THỰC TẾ VS AI (THÁNG ĐÃ QUA) ---
                st.subheader("🔢 Actual vs AI Forecast Variance (2026)")
                act_2026 = df_monthly[df_monthly['ds'].dt.year == 2026].copy()
                # Merge thực tế với dự báo AI dựa trên cột ngày 'ds'
                comparison = pd.merge(act_2026, fcst_26[['ds', 'yhat']], on='ds', how='left')
                
                if not comparison.empty:
                    comparison['Variance (%)'] = ((comparison['y'] - comparison['yhat']) / comparison['yhat']) * 100
                    
                    # Định dạng bảng hiển thị
                    st.dataframe(comparison.style.format({
                        'ds': lambda x: x.strftime('%m/%Y'),
                        'y': '{:,.0f}',
                        'yhat': '{:,.0f}',
                        'Variance (%)': '{:+.1f}%'
                    }), use_container_width=True)
                    
                    # Lời khuyên dựa trên kết quả so sánh
                    last_month_var = comparison['Variance (%)'].iloc[-1]
                    st.markdown("### 💡 AI Analytical Advice")
                    if last_month_var > 15:
                        st.warning(f"Cảnh báo: Tháng gần nhất vượt dự báo **{last_month_var:.1f}%**. Jenna hãy kiểm tra ngay tiến độ giao linh kiện từ Supplier để tránh hụt hàng (shortage).")
                    elif last_month_var < -15:
                        st.error(f"Cảnh báo: Nhu cầu thực tế thấp hơn dự báo **{abs(last_month_var):.1f}%**. Cần cân nhắc điều chỉnh giảm Forecast các tháng tới để tối ưu tồn kho.")
                    else:
                        st.success("Tình hình kinh doanh đang bám rất sát dự báo AI. Tiếp tục duy trì kế hoạch hiện tại.")

            with tab2:
                # (Phần Tab 2 Strategic Plan giữ nguyên logic như cũ)
                st.subheader("📋 2026 Strategic Plan")
                # ... [Code logic Tab 2 giống bản trước] ...
