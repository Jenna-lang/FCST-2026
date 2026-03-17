import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Phân tích Pareto & Khách hàng 2026", layout="wide")
st.title("🚀 Hệ Thống Phân Tích Cung Ứng Pareto 2026")

@st.cache_data
def load_data():
    df = pd.read_excel('AICheck.xlsx')
    df['ds'] = pd.to_datetime(df['Requested deliv. date'])
    return df

try:
    df = load_data()
    
    # Lọc danh sách Pareto (85% giá trị)
    summary = df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
    summary['Cum_Pct'] = summary['M USD'].cumsum() / summary['M USD'].sum()
    pareto_list = summary[summary['Cum_Pct'] <= 0.85]['Material name'].unique()

    st.sidebar.header("🔍 Bộ Lọc Truy Vấn")
    selected_prod = st.sidebar.selectbox("Chọn mã linh kiện Pareto:", pareto_list)

    if selected_prod:
        # Lấy dữ liệu của toàn bộ mã hàng đó
        prod_df = df[df['Material name'] == selected_prod].copy()
        
        # TẠO CÁC TAB HIỂN THỊ
        tab1, tab2 = st.tabs(["📊 Tổng quan mã hàng (Tất cả Khách)", "🎯 Chi tiết dự báo từng Khách"])

        with tab1:
            st.subheader(f"Phân tích tỷ trọng khách hàng cho: {selected_prod}")
            col_a, col_b = st.columns(2)
            
            with col_a:
                # Biểu đồ tròn: Tỷ trọng doanh số theo khách hàng
                cust_share = prod_df.groupby('End Cust')['M USD'].sum().reset_index()
                fig_pie = px.pie(cust_share, values='M USD', names='End Cust', 
                                 title="Tỷ trọng doanh số (M USD) giữa các khách hàng",
                                 hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_b:
                # Biểu đồ cột: Lịch sử đặt hàng phân theo khách hàng
                prod_df['Month'] = prod_df['ds'].dt.to_period('M').dt.to_timestamp()
                monthly_cust = prod_df.groupby(['Month', 'End Cust'])['Order qty.(A)'].sum().reset_index()
                fig_bar = px.bar(monthly_cust, x='Month', y='Order qty.(A)', color='End Cust',
                                 title="Lịch sử đặt hàng theo tháng phân loại khách hàng")
                st.plotly_chart(fig_bar, use_container_width=True)

        with tab2:
            st.subheader("Dự báo AI cho từng khách hàng cụ thể")
            customers_of_prod = sorted(prod_df['End Cust'].unique())
            selected_cust = st.selectbox("Chọn khách hàng cần xem dự báo:", customers_of_prod)

            if selected_cust:
                # Lọc dữ liệu cho khách hàng cụ thể
                cust_data = prod_df[prod_df['End Cust'] == selected_cust].copy()
                cust_data['ds_month'] = cust_data['ds'].dt.to_period('M').dt.to_timestamp()
                p_df = cust_data.groupby('ds_month')['Order qty.(A)'].sum().reset_index().rename(columns={'ds_month':'ds', 'Order qty.(A)':'y'})

                if len(p_df) >= 2:
                    m = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
                    m.fit(p_df)
                    future = m.make_future_dataframe(periods=12, freq='MS')
                    forecast = m.predict(future)
                    for col in ['yhat', 'yhat_lower', 'yhat_upper']: forecast[col] = forecast[col].clip(lower=0)

                    # Giao diện kết quả
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.pyplot(m.plot(forecast))
                    with c2:
                        st.info(f"💡 **Lời khuyên cho {selected_cust}:**")
                        f_2026 = forecast[forecast['ds'].dt.year == 2026]
                        st.write("- Theo dõi kỹ xu hướng tháng tới để điều chỉnh tồn kho.")
                        st.metric("Dự báo tháng tới", f"{f_2026['yhat'].iloc[0]:,.0f} Pcs")

                    # Bảng dữ liệu 2026
                    st.write(f"**Bảng số liệu dự báo 2026 cho {selected_cust}:**")
                    res_table = f_2026[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                    res_table.columns = ['Tháng', 'Trung bình', 'Min', 'Max']
                    res_table['Tháng'] = res_table['Tháng'].dt.strftime('%m/%Y')
                    st.dataframe(res_table.style.format('{:,.0f}', subset=['Trung bình', 'Min', 'Max']), use_container_width=True)
                else:
                    st.warning("⚠️ Không đủ dữ liệu (ít hơn 2 tháng) để AI dự báo cho khách hàng này.")

except Exception as e:
    st.error(f"Đã xảy ra lỗi: {e}")
