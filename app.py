import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Dự báo Pareto 2026", layout="wide")
st.title("🚀 Hệ Thống Phân Tích Cung Ứng Pareto 2026")

@st.cache_data
def load_data():
    df = pd.read_excel('AICheck.xlsx')
    df.columns = [str(col).strip() for col in df.columns]
    df['ds'] = pd.to_datetime(df['Requested deliv. date'])
    return df

try:
    df = load_data()
    
    # --- LOGIC TÌM CỘT TÊN KHÁCH HÀNG THÔNG MINH ---
    # Ưu tiên các cột có chữ 'Cust' hoặc 'Name' NHƯNG không chứa chữ 'Code' hoặc 'ID'
    potential_name_cols = [col for col in df.columns if ('cust' in col.lower() or 'name' in col.lower()) 
                          and 'code' not in col.lower() and 'id' not in col.lower()]
    
    if potential_name_cols:
        cust_col = potential_name_cols[0]
    else:
        # Nếu không tìm thấy, liệt kê tất cả cột để bạn dễ kiểm tra lỗi
        cust_col = df.columns[0] 

    # Lọc Pareto (85%)
    summary = df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
    summary['Cum_Pct'] = summary['M USD'].cumsum() / summary['M USD'].sum()
    pareto_list = summary[summary['Cum_Pct'] <= 0.85]['Material name'].unique()

    st.sidebar.header("🔍 Bộ Lọc Truy Vấn")
    selected_prod = st.sidebar.selectbox("1. Chọn mã linh kiện Pareto:", pareto_list)

    if selected_prod:
        prod_df = df[df['Material name'] == selected_prod].copy()
        tab1, tab2 = st.tabs(["📊 Tổng quan mã hàng", "🎯 Dự báo & Lời khuyên AI"])

        with tab1:
            st.subheader(f"Phân tích khách hàng cho: {selected_prod}")
            c_a, c_b = st.columns(2)
            with c_a:
                fig_pie = px.pie(prod_df.groupby(cust_col)['M USD'].sum().reset_index(), 
                                 values='M USD', names=cust_col, hole=0.4, title=f"Tỷ trọng theo: {cust_col}")
                st.plotly_chart(fig_pie, use_container_width=True)
            with c_b:
                prod_df['Month'] = prod_df['ds'].dt.to_period('M').dt.to_timestamp()
                fig_bar = px.bar(prod_df.groupby(['Month', cust_col])['Order qty.(A)'].sum().reset_index(), 
                                 x='Month', y='Order qty.(A)', color=cust_col, title="Lịch sử đặt hàng")
                st.plotly_chart(fig_bar, use_container_width=True)

        with tab2:
            # Sắp xếp danh sách tên khách hàng cho đẹp
            customers = sorted([str(x) for x in prod_df[cust_col].unique()])
            selected_cust = st.selectbox(f"Chọn {cust_col} cần xem chi tiết:", customers)
            
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
                        st.subheader("🤖 AI Supply Chain Advisor")
                        f_2026 = forecast[forecast['ds'].dt.year == 2026]
                        avg_v = f_2026['yhat'].mean()
                        st.metric("Nhu cầu dự kiến 2026 (TB/Tháng)", f"{avg_v:,.0f} Pcs")
                        
                        # Lời khuyên tự động
                        if avg_v > p_df['y'].mean() * 1.15:
                            st.error(f"⚠️ **Cảnh báo:** {selected_cust} đang có xu hướng tăng đơn mạnh (>15%). Hãy liên hệ Sales để xác nhận slot sản xuất.")
                        else:
                            st.success(f"✅ **Ổn định:** Nhu cầu của {selected_cust} trong năm 2026 ở mức bình thường.")

                    st.subheader(f"📋 Bảng số liệu chi tiết 2026 cho {selected_cust}")
                    res = f_2026[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                    res.columns = ['Tháng', 'Dự báo TB', 'Min (An toàn)', 'Max (Kỳ vọng)']
                    res['Tháng'] = res['Tháng'].dt.strftime('%m/%Y')
                    st.dataframe(res.style.format('{:,.0f}', subset=['Dự báo TB', 'Min (An toàn)', 'Max (Kỳ vọng)']), use_container_width=True)
                else:
                    st.warning(f"⚠️ Dữ liệu của {selected_cust} quá ít để AI dự báo.")

except Exception as e:
    st.error(f"Đã xảy ra lỗi: {e}")
