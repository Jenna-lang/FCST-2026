import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. Cấu hình & Cache dữ liệu
st.set_page_config(page_title="AI Supply Chain Advisor", layout="wide")

@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_excel('AICheck.xlsx')
        df.columns = [str(col).strip() for col in df.columns]
        # Chuyển đổi ngày tháng an toàn
        df['ds'] = pd.to_datetime(df['Requested deliv. date'], errors='coerce')
        df = df.dropna(subset=['ds'])
        return df
    except Exception as e:
        st.error(f"Lỗi tải file: {e}")
        return None

@st.cache_resource
def fast_forecast(data_series):
    if len(data_series) < 2: return None
    try:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(data_series)
        future = m.make_future_dataframe(periods=12, freq='MS')
        forecast = m.predict(future)
        return forecast[['ds', 'yhat']]
    except:
        return None

df = load_data()

# Khởi tạo các biến mặc định để tránh NameError
selected_cust = None

if df is not None:
    st.sidebar.header("⚡ Fast Config")
    # Tự động tìm cột Customer và CIE
    possible_cust_cols = [c for c in df.columns if 'Customer' in c or 'Cust' in c]
    cust_col = st.sidebar.selectbox("Customer Column:", possible_cust_cols if possible_cust_cols else df.columns)
    
    possible_cie_cols = [c for c in df.columns if 'CIE' in c]
    cie_col = st.sidebar.selectbox("CIE Column:", possible_cie_cols if possible_cie_cols else df.columns)
    
    cust_list = sorted(df[cust_col].unique().astype(str))
    selected_cust = st.sidebar.selectbox("1. Select Customer:", ["-- Chọn khách hàng --"] + cust_list)

    if selected_cust and selected_cust != "-- Chọn khách hàng --":
        cust_df = df[df[cust_col].astype(str) == selected_cust].copy()
        
        # Pareto calculation
        sales = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        sales['Cum_Pct'] = sales['M USD'].cumsum() / (sales['M USD'].sum() if sales['M USD'].sum() > 0 else 1)
        top_prods = sales[sales['Cum_Pct'] <= 0.81]['Material name'].unique()
        
        selected_prod = st.selectbox("2. Select Product:", top_prods if len(top_prods) > 0 else df['Material name'].unique()[:5])
        
        tab1, tab2 = st.tabs(["📊 Yearly Comparison", "📋 Full Pareto Details"])

        with tab1:
            prod_df = cust_df[cust_df['Material name'] == selected_prod].copy()
            prod_df['m'] = prod_df['ds'].dt.to_period('M').dt.to_timestamp()
            actual_prod = prod_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
            
            res = fast_forecast(actual_prod)
            
            if res is not None:
                # Tính toán tổng sản lượng
                sum_25 = actual_prod[actual_prod['ds'].dt.year == 2025]['y'].sum()
                sum_26 = res[res['ds'].dt.year == 2026]['yhat'].sum()
                growth = ((sum_26 - sum_25) / sum_25 * 100) if sum_25 > 0 else 0
                
                # Sửa lỗi: Luôn định nghĩa màu sắc rõ ràng
                bar_color = "green" if growth >= 0 else "red"
                st.markdown(f"### Yearly Growth: :{bar_color}[{growth:.1f}%]")
                
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig_l = go.Figure()
                    fig_l.add_trace(go.Scatter(x=actual_prod['ds'], y=actual_prod['y'], name="Actual"))
                    f_only = res[res['ds'] > actual_prod['ds'].max()]
                    fig_l.add_trace(go.Scatter(x=f_only['ds'], y=f_only['yhat'], line=dict(dash='dash', color=bar_color), name="2026 FCST"))
                    st.plotly_chart(fig_l, use_container_width=True)
                
                with c2:
                    # Sửa lỗi ValueError: Đảm bảo giá trị truyền vào Bar chart là hợp lệ
                    fig_b = go.Figure(go.Bar(
                        x=['2025 (Actual)', '2026 (Forecast)'], 
                        y=[sum_25, sum_26], 
                        marker_color=['#1f77b4', bar_color]
                    ))
                    fig_b.update_layout(title="Volume Comparison")
                    st.plotly_chart(fig_b, use_container_width=True)
            else:
                st.warning("Không đủ dữ liệu để dự báo sản phẩm này.")

       with tab2:
            st.subheader("📋 Bảng tổng hợp Pareto 2026 (Xử lý thần tốc)")
            
            # 1. Lấy tỷ lệ tăng trưởng từ Tab 1 để áp dụng cho Tab 2
            global_growth_factor = 1.0
            # Kiểm tra nếu biến 'growth' tồn tại từ Tab 1
            if 'growth' in locals():
                global_growth_factor = 1 + (growth / 100)

            # 2. Xử lý dữ liệu nhanh bằng Groupby
            # Lọc các sản phẩm thuộc nhóm Pareto
            pareto_df = cust_df[cust_df['Material name'].isin(top_prods)].copy()
            
            # Tính trung bình sản lượng thực tế năm 2025 làm cơ sở
            df_2025 = pareto_df[pareto_df['ds'].dt.year == 2025]
            if not df_2025.empty:
                avg_stats = df_2025.groupby(['Material name', cie_col])['Order qty.(A)'].mean().reset_index()
                
                all_rows = []
                # Tạo danh sách các tháng trong năm 2026
                months_2026 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')

                for _, row in avg_stats.iterrows():
                    base_qty = row['Order qty.(A)']
                    for m_date in months_2026:
                        all_rows.append({
                            'Month': m_date.strftime('%m/%Y'),
                            'Product': row['Material name'],
                            'CIE': row[cie_col],
                            'Qty (Pcs)': round(base_qty * global_growth_factor, 0)
                        })
                
                if all_rows:
                    final_table = pd.DataFrame(all_rows)
                    
                    # Bộ lọc nhanh (Quick Filters)
                    c_f1, c_f2 = st.columns(2)
                    with c_f1:
                        f_p = st.multiselect("Lọc Sản phẩm:", top_prods)
                    with c_f2:
                        f_m = st.multiselect("Lọc Tháng:", final_table['Month'].unique())
                    
                    # Áp dụng logic lọc
                    df_view = final_table.copy()
                    if f_p: 
                        df_view = df_view[df_view['Product'].isin(f_p)]
                    if f_m: 
                        df_view = df_view[df_view['Month'].isin(f_m)]
                    
                    # Hiển thị bảng số liệu
                    st.dataframe(df_view.style.format("{:,.0f}", subset=['Qty (Pcs)']), use_container_width=True)
                    
                    # Nút tải báo cáo CSV
                    st.download_button(
                        label="📥 Tải báo cáo Pareto 2026 (CSV)",
                        data=df_view.to_csv(index=False).encode('utf-8-sig'),
                        file_name=f"Pareto_Fast_FCST_{selected_cust}.csv",
                        mime='text/csv'
                    )
            else:
                st.warning("Không tìm thấy dữ liệu năm 2025 của khách hàng này để tính toán nhanh.")
