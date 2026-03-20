import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. Cấu hình trang
st.set_page_config(page_title="AI Supply Chain Advisor", layout="wide")
st.title("🤖 AI Strategic Advisor: 2026 Forecast")

@st.cache_data
def load_data():
    try:
        df = pd.read_excel('AICheck.xlsx')
        df.columns = [str(col).strip() for col in df.columns]
        df['ds'] = pd.to_datetime(df['Requested deliv. date'])
        return df
    except Exception as e:
        st.error(f"Lỗi tải file Excel: {e}")
        return None

df = load_data()

if df is not None:
    # 2. Cấu hình Sidebar
    st.sidebar.header("Thiết lập dữ liệu")
    cust_col = st.sidebar.selectbox("Cột Tên Khách hàng:", df.columns)
    cie_col = st.sidebar.selectbox("Cột Mã màu CIE:", df.columns)
    
    # Lấy danh sách khách hàng
    cust_list = sorted(df[cust_col].unique().astype(str))
    selected_cust = st.sidebar.selectbox("1. Chọn Khách hàng:", cust_list)

    if selected_cust:
        # Lọc dữ liệu theo khách hàng đã chọn
        cust_df = df[df[cust_col].astype(str) == selected_cust].copy()
        
        # --- PHÂN TÍCH PARETO (80/20) THEO TỪNG KHÁCH HÀNG ---
        sales = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        sales['Cum_Pct'] = sales['M USD'].cumsum() / sales['M USD'].sum()
        top_prods = sales[sales['Cum_Pct'] <= 0.81]['Material name'].unique()
        
        st.sidebar.success(f"Tìm thấy {len(top_prods)} sản phẩm chủ lực (80% doanh thu).")
        selected_prod = st.selectbox("2. Chọn sản phẩm Pareto:", top_prods)

        if selected_prod:
            prod_df = cust_df[cust_df['Material name'] == selected_prod].copy()
            cie_options = sorted(prod_df[cie_col].unique().astype(str))
            selected_cies = st.multiselect("3. Chọn mã màu CIE:", cie_options, default=cie_options[:1])

            if selected_cies:
                fig = go.Figure()
                forecast_results = []
                ai_insights = []
                
                for cie in selected_cies:
                    cie_df = prod_df[prod_df[cie_col].astype(str) == cie].copy()
                    cie_df['m'] = cie_df['ds'].dt.to_period('M').dt.to_timestamp()
                    actual = cie_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
                    
                    if len(actual) >= 2:
                        # Dự báo AI với Prophet
                        m = Prophet(yearly_seasonality=True).fit(actual)
                        future = m.make_future_dataframe(periods=12, freq='MS')
                        fcst = m.predict(future)
                        fcst['yhat'] = fcst['yhat'].clip(lower=0)
                        
                        # Vẽ biểu đồ lịch sử và dự báo
                        fig.add_trace(go.Scatter(x=actual['ds'], y=actual['y'], mode='lines+markers', name=f"Thực tế - {cie}"))
                        f_only = fcst[fcst['ds'] > actual['ds'].max()]
                        fig.add_trace(go.Scatter(x=f_only['ds'], y=f_only['yhat'], mode='lines', line=dict(dash='dash'), name=f"Dự báo - {cie}"))
                        
                        # Nhận xét từ AI Advisor
                        avg_past = actual['y'].tail(12).mean()
                        max_fcst = f_only[f_only['ds'].dt.year == 2026]['yhat'].max()
                        growth = ((max_fcst - avg_past) / avg_past * 100) if avg_past > 0 else 0
                        status = "Tăng trưởng" if growth > 5 else "Giảm" if growth < -5 else "Ổn định"
                        ai_insights.append(f"**Màu {cie}**: Xu hướng **{status}** ({growth:.1f}% so với 12 tháng qua).")
                        
                        # Chuẩn bị dữ liệu bảng
                        f2026 = fcst[fcst['ds'].dt.year == 2026].copy()
                        f2026['Month'] = f2026['ds'].dt.strftime('%m/%Y')
                        f2026['CIE'] = cie
                        f2026['Customer'] = selected_cust
                        f2026['Product'] = selected_prod
                        forecast_results.append(f2026[['Month', 'Product', 'CIE', 'Customer', 'yhat']])

                # 4. Hiển thị Giao diện bằng TABS
                tab1, tab2 = st.tabs(["📊 Biểu đồ phân tích (Chart)", "📋 Bảng dữ liệu chi tiết (FCST Details)"])

                with tab1:
                    st.subheader("💡 AI Advisor Insights")
                    for insight in ai_insights:
                        st.info(insight)
                    st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    if forecast_results:
                        st.subheader(f"Dự báo chi tiết năm 2026 cho {selected_prod}")
                        final_table = pd.concat(forecast_results)
                        # Định dạng bảng chuyên nghiệp
                        display_df = final_table.rename(columns={
                            'Month': 'Tháng', 
                            'Product': 'Tên sản phẩm', 
                            'CIE': 'Mã màu CIE',
                            'Customer': 'Khách hàng',
                            'yhat': 'Số lượng dự báo (Pcs)'
                        })
                        st.dataframe(display_df.style.format("{:,.0f}", subset=['Số lượng dự báo (Pcs)']), use_container_width=True)
                        
                        # Nút tải dữ liệu
                        csv = display_df.to_csv(index=False).encode('utf-8-sig')
                        st.download_button("📥 Tải bảng dự báo (CSV)", data=csv, file_name=f"FCST_2026_{selected_cust}.csv", mime="text/csv")
            else:
                st.warning("Vui lòng chọn ít nhất một mã màu CIE.")
