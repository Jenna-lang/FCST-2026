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
        st.error(f"Lỗi tải file: {e}")
        return None

df = load_data()

if df is not None:
    # 2. Sidebar
    st.sidebar.header("Thiết lập")
    cust_col = st.sidebar.selectbox("Cột Khách hàng:", df.columns)
    cie_col = st.sidebar.selectbox("Cột Mã màu CIE:", df.columns)
    
    cust_list = sorted(df[cust_col].unique().astype(str))
    selected_cust = st.sidebar.selectbox("1. Chọn Khách hàng:", cust_list)

    if selected_cust:
        cust_df = df[df[cust_col].astype(str) == selected_cust].copy()
        
        # --- LOGIC PARETO 80/20 CHO TỪNG KHÁCH HÀNG ---
        sales = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        sales['Cum_Pct'] = sales['M USD'].cumsum() / sales['M USD'].sum()
        top_prods = sales[sales['Cum_Pct'] <= 0.81]['Material name'].unique()
        
        # 3. Lựa chọn xem biểu đồ (Chỉ xem 1 sản phẩm để tránh rối mắt)
        selected_prod_for_chart = st.selectbox("2. Chọn sản phẩm để xem biểu đồ chi tiết:", top_prods)
        
        # Danh sách lưu kết quả bảng tổng hợp và biểu đồ
        all_pareto_forecasts = []
        chart_data = []
        ai_insights = []

        # --- CHẠY DỰ BÁO TỰ ĐỘNG CHO TOÀN BỘ DANH MỤC PARETO ---
        with st.spinner(f'Đang tính toán dự báo cho {len(top_prods)} sản phẩm Pareto...'):
            for prod in top_prods:
                temp_prod_df = cust_df[cust_df['Material name'] == prod].copy()
                for cie in temp_prod_df[cie_col].unique():
                    cie_df = temp_prod_df[temp_prod_df[cie_col].astype(str) == cie].copy()
                    cie_df['m'] = cie_df['ds'].dt.to_period('M').dt.to_timestamp()
                    actual = cie_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
                    
                    if len(actual) >= 2:
                        m = Prophet(yearly_seasonality=True).fit(actual)
                        future = m.make_future_dataframe(periods=12, freq='MS')
                        fcst = m.predict(future)
                        fcst['yhat'] = fcst['yhat'].clip(lower=0)
                        
                        # Lưu dữ liệu cho BẢNG CHI TIẾT
                        f2026 = fcst[fcst['ds'].dt.year == 2026].copy()
                        f2026['Month'] = f2026['ds'].dt.strftime('%m/%Y')
                        f2026['Product'] = prod
                        f2026['CIE'] = cie
                        all_pareto_forecasts.append(f2026[['Month', 'Product', 'CIE', 'yhat']])
                        
                        # Lưu dữ liệu cho BIỂU ĐỒ (Nếu là sản phẩm đang chọn)
                        if prod == selected_prod_for_chart:
                            chart_data.append({'cie': cie, 'actual': actual, 'fcst': fcst})
                            # Nhận xét AI cho sản phẩm đang xem
                            avg = actual['y'].tail(12).mean()
                            f_max = f2026['yhat'].max()
                            growth = ((f_max - avg) / avg * 100) if avg > 0 else 0
                            ai_insights.append(f"**{cie}**: {'Tăng' if growth > 5 else 'Giảm' if growth < -5 else 'Ổn định'} ({growth:.1f}%).")

        # --- HIỂN THỊ KẾT QUẢ THEO TABS ---
        tab1, tab2 = st.tabs(["📊 Analytics Chart", "📋 All Pareto FCST Details"])

        with tab1:
            st.subheader(f"Xu hướng sản phẩm: {selected_prod_for_chart}")
            fig = go.Figure()
            for item in chart_data:
                # Vẽ thực tế (Solid) và Dự báo (Dashed)
                fig.add_trace(go.Scatter(x=item['actual']['ds'], y=item['actual']['y'], name=f"Actual {item['cie']}"))
                f_only = item['fcst'][item['fcst']['ds'] > item['actual']['ds'].max()]
                fig.add_trace(go.Scatter(x=f_only['ds'], y=f_only['yhat'], line=dict(dash='dash'), name=f"FCST {item['cie']}"))
            st.plotly_chart(fig, use_container_width=True)
            for insight in ai_insights: st.info(insight)

        with tab2:
            if all_pareto_forecasts:
                st.subheader(f"Bảng tổng hợp dự báo Pareto 2026 - {selected_cust}")
                final_df = pd.concat(all_pareto_forecasts)
                # Hiển thị tất cả mã hàng Pareto
                st.dataframe(final_df.rename(columns={'yhat': 'Qty (Pcs)'}).style.format("{:,.0f}", subset=['Qty (Pcs)']), use_container_width=True)
                
                csv = final_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 Tải toàn bộ Pareto (CSV)", data=csv, file_name=f"Full_Pareto_{selected_cust}.csv")
