import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Hệ thống Dự báo Pareto 2026", layout="wide")
st.title("🚀 Hệ Thống Phân Tích Cung Ứng Pareto 2026")

@st.cache_data
def load_data():
    df = pd.read_excel('AICheck.xlsx')
    # Tự động chuẩn hóa tên cột (xóa khoảng trắng, viết thường)
    df.columns = [str(col).strip() for col in df.columns]
    df['ds'] = pd.to_datetime(df['Requested deliv. date'])
    return df

try:
    df = load_data()
    
    # Tự động tìm cột khách hàng (đề phòng tên cột là 'End Cust', 'Customer', 'Cust Name'...)
    cust_col = next((col for col in df.columns if 'cust' in col.lower() or 'khách' in col.lower()), None)
    
    if not cust_col:
        st.error("❌ Không tìm thấy cột 'End Cust' hoặc 'Customer' trong file Excel. Vui lòng kiểm tra lại tên cột.")
    else:
        # Lọc Pareto (85%)
        summary = df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        summary['Cum_Pct'] = summary['M USD'].cumsum() / summary['M USD'].sum()
        pareto_list = summary[summary['Cum_Pct'] <= 0.85]['Material name'].unique()

        st.sidebar.header("🔍 Bộ Lọc Truy Vấn")
        selected_prod = st.sidebar.selectbox("Chọn mã linh kiện Pareto:", pareto_list)

        if selected_prod:
            prod_df = df[df['Material name'] == selected_prod].copy()
            tab1, tab2 = st.tabs(["📊 Tổng quan mã hàng", "🎯 Dự báo chi tiết khách hàng"])

            with tab1:
                st.subheader(f"Phân tích tỷ trọng khách hàng cho: {selected_prod}")
                col_a, col_b = st.columns(2)
                with col_a:
                    cust_share = prod_df.groupby(cust_col)['M USD'].sum().reset_index()
                    fig_pie = px.pie(cust_share, values='M USD', names=cust_col, hole=0.4, title="Tỷ trọng doanh số")
                    st.plotly_chart(fig_pie, use_container_width=True)
                with col_b:
                    prod_df['Month'] = prod_df['ds'].dt.to_period('M').dt.to_timestamp()
                    monthly_cust = prod_df.groupby(['Month', cust_col])['Order qty.(A)'].sum().reset_index()
                    fig_bar = px.bar(monthly_cust, x='Month', y='Order qty.(A)', color=cust_col, title="Lịch sử đặt hàng")
                    st.plotly_chart(fig_bar, use_container_width=True)

            with tab2:
                customers = sorted(prod_df[cust_col].unique())
                selected_cust = st.selectbox("Chọn khách hàng:", customers)
                if selected_cust:
                    cust_data = prod_df[prod_df[cust_col] == selected_cust].copy()
                    cust_data['ds_month'] = cust_data['ds'].dt.to_period('M').dt.to_timestamp()
                    p_df = cust_data.groupby('ds_month')['Order qty.(A)'].sum().reset_index().rename(columns={'ds_month':'ds', 'Order qty.(A)':'y'})

                    if len(p_df) >= 2:
                        m = Prophet(yearly_seasonality=True).fit(p_df)
                        future = m.make_future_dataframe(periods=12, freq='MS')
                        forecast = m.predict(future)
                        st.pyplot(m.plot(forecast))
                        
                        f_2026 = forecast[forecast['ds'].dt.year == 2026]
                        st.subheader(f"📋 Kế hoạch 2026 cho {selected_cust}")
                        res = f_2026[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                        res.columns = ['Tháng', 'Trung bình', 'Min', 'Max']
                        res['Tháng'] = res['Tháng'].dt.strftime('%m/%Y')
                        st.dataframe(res.style.format('{:,.0f}', subset=['Trung bình', 'Min', 'Max']), use_container_width=True)
except Exception as e:
    st.error(f"Đã xảy ra lỗi: {e}")
