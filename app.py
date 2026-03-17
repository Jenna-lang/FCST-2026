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
    df.columns = [str(col).strip() for col in df.columns]
    df['ds'] = pd.to_datetime(df['Requested deliv. date'])
    return df

try:
    df = load_data()
    # Tìm cột Tên khách hàng (không phải mã code)
    cust_col = next((col for col in df.columns if 'cust' in col.lower() and 'code' not in col.lower()), "End Cust")

    summary = df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
    summary['Cum_Pct'] = summary['M USD'].cumsum() / summary['M USD'].sum()
    pareto_list = summary[summary['Cum_Pct'] <= 0.85]['Material name'].unique()

    selected_prod = st.sidebar.selectbox("1. Chọn mã linh kiện Pareto:", pareto_list)

    if selected_prod:
        prod_df = df[df['Material name'] == selected_prod].copy()
        tab1, tab2 = st.tabs(["📊 Tổng quan mã hàng", "🎯 Dự báo & Lời khuyên AI"])

        with tab1:
            st.subheader(f"Phân tích khách hàng cho: {selected_prod}")
            c_a, c_b = st.columns(2)
            with c_a:
                fig_pie = px.pie(prod_df.groupby(cust_col)['M USD'].sum().reset_index(), values='M USD', names=cust_col, hole=0.4, title="Tỷ trọng doanh số")
                st.plotly_chart(fig_pie, use_container_width=True)
            with c_b:
                prod_df['Month'] = prod_df['ds'].dt.to_period('M').dt.to_timestamp()
                fig_bar = px.bar(prod_df.groupby(['Month', cust_col])['Order qty.(A)'].sum().reset_index(), x='Month', y='Order qty.(A)', color=cust_col, title="Lịch sử đặt hàng")
                st.plotly_chart(fig_bar, use_container_width=True)

        with tab2:
            customers = sorted(prod_df[cust_col].unique())
            selected_cust = st.selectbox("Chọn Tên Khách Hàng:", customers)
            if selected_cust:
                cust_data = prod_df[prod_df[cust_col] == selected_cust].copy()
                cust_data['ds_month'] = cust_data['ds'].dt.to_period('M').dt.to_timestamp()
                p_df = cust_data.groupby('ds_month')['Order qty.(A)'].sum().reset_index().rename(columns={'ds_month':'ds', 'Order qty.(A)':'y'})

                if len(p_df) >= 2:
                    m = Prophet(yearly_seasonality=True).fit(p_df)
                    future = m.make_future_dataframe(periods=12, freq='MS')
                    forecast = m.predict(future)
                    
                    st.divider()
                    l_col, r_col = st.columns([2, 1])
                    with l_col:
                        st.pyplot(m.plot(forecast))
                    with r_col:
                        st.subheader("🤖 AI Advisor")
                        f_2026 = forecast[forecast['ds'].dt.year == 2026]
                        avg_v = f_2026['yhat'].mean()
                        st.metric("Dự báo TB 2026", f"{avg_v:,.0f} Pcs")
                        if avg_v > p_df['y'].mean() * 1.1:
                            st.error("🚩 Xu hướng tăng! Cần đàm phán sớm với Supplier.")
                        else:
                            st.success("✅ Xu hướng ổn định.")

                    st.subheader(f"📋 Kế hoạch 2026 cho {selected_cust}")
                    res = f_2026[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                    res.columns = ['Tháng', 'Trung bình', 'Min', 'Max']
                    res['Tháng'] = res['Tháng'].dt.strftime('%m/%Y')
                    st.dataframe(res.style.format('{:,.0f}', subset=['Trung bình', 'Min', 'Max']), use_container_width=True)
except Exception as e:
    st.error(f"Đã xảy ra lỗi: {e}")
