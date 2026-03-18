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
    # Loại bỏ khoảng trắng thừa trong tên cột
    df.columns = [str(col).strip() for col in df.columns]
    df['ds'] = pd.to_datetime(df['Requested deliv. date'])
    return df

try:
    df = load_data()
    
    # --- CẤU HÌNH SIDEBAR ---
    st.sidebar.header("⚙️ Cấu hình dữ liệu")
    # Cho phép chọn cột tên khách hàng từ thực tế file Excel
    cust_col = st.sidebar.selectbox("Chọn cột Tên Khách Hàng:", df.columns)

    # Lọc Pareto (85% doanh số)
    summary = df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
    summary['Cum_Pct'] = summary['M USD'].cumsum() / summary['M USD'].sum()
    pareto_list = summary[summary['Cum_Pct'] <= 0.85]['Material name'].unique()

    st.sidebar.divider()
    selected_prod = st.sidebar.selectbox("Chọn mã linh kiện Pareto:", pareto_list)

    if selected_prod:
        prod_df = df[df['Material name'] == selected_prod].copy()
        tab1, tab2 = st.tabs(["📊 Tổng quan mã hàng", "🎯 Dự báo & Lời khuyên AI 2026"])

        with tab1:
            st.subheader(f"Phân tích tỷ trọng khách hàng cho: {selected_prod}")
            c1, c2 = st.columns(2)
            with c1:
                fig_p = px.pie(prod_df.groupby(cust_col)['M USD'].sum().reset_index(), 
                               values='M USD', names=cust_col, hole=0.4)
                st.plotly_chart(fig_p, use_container_width=True)
            with c2:
                prod_df['Month'] = prod_df['ds'].dt.to_period('M').dt.to_timestamp()
                fig_b = px.bar(prod_df.groupby(['Month', cust_col])['Order qty.(A)'].sum().reset_index(), 
                               x='Month', y='Order qty.(A)', color=cust_col)
                st.plotly_chart(fig_b, use_container_width=True)

        with tab2:
            # --- TÍNH NĂNG XEM TỔNG QUAN (ALL) ---
            cust_list = sorted([str(x) for x in prod_df[cust_col].unique()])
            options = ["ALL (Tổng hợp tất cả khách hàng)"] + cust_list
            selected_cust = st.selectbox("Chọn khách hàng cụ thể hoặc xem Tổng quát:", options)
            
            if selected_cust == "ALL (Tổng hợp tất cả khách hàng)":
                cust_data = prod_df.copy()
                label = "TẤT CẢ KHÁCH HÀNG"
            else:
                cust_data = prod_df[prod_df[cust_col] == selected_cust].copy()
                label = selected_cust

            # Chuẩn bị dữ liệu cho AI
            cust_data['ds_month'] = cust_data['ds'].dt.to_period('M').dt.to_timestamp()
            p_df = cust_data.groupby('ds_month')['Order qty.(A)'].sum().reset_index().rename(columns={'ds_month':'ds', 'Order qty.(A)':'y'})

            if len(p_df) >= 2:
                # Chạy AI dự báo
                m = Prophet(yearly_seasonality=True).fit(p_df)
                future = m.make_future_dataframe(periods=12, freq='MS')
                forecast = m.predict(future)
                for c in ['yhat', 'yhat_lower', 'yhat_upper']: forecast[c] = forecast[c].clip(lower=0)
                
                st.divider()
                col_l, col_r = st.columns([2, 1])
                
                with col_l:
                    st.subheader(f"📈 Biểu đồ dự báo nhu cầu: {label}")
                    st.pyplot(m.plot(forecast))
                
                with col_r:
                    st.subheader("🤖 AI Supply Chain Advisor")
                    f_2026 = forecast[forecast['ds'].dt.year == 2026]
                    avg_2026 = f_2026['yhat'].mean()
                    st.metric("Nhu cầu TB tháng 2026", f"{avg_2026:,.0f} Pcs")
                    
                    # Cảnh báo dựa trên xu hướng
                    hist_avg = p_df['y'].mean()
                    growth = (avg_2026 / hist_avg - 1) if hist_avg > 0 else 0
                    
                    if growth > 0.15:
                        st.error(f"🚩 **Xu hướng tăng:** Dự báo tăng {growth:.1%}. Đề xuất chuẩn bị nguồn cung dự phòng.")
                    elif growth < -0.15:
                        st.warning(f"📉 **Xu hướng giảm:** Dự báo giảm {abs(growth):.1%}. Cần kiểm tra lại kế hoạch mua hàng.")
                    else:
                        st.success("✅ **Ổn định:** Nhu cầu năm 2026 dự kiến tương đương mức hiện tại.")
                    
                    st.info(f"💡 **Mức dự phòng an toàn:** {f_2026['yhat_upper'].max():,.0f} Pcs")

                st.subheader(f"📋 Bảng số liệu chi tiết 2026 cho {label}")
                res = f_2026[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                res.columns = ['Tháng', 'Trung bình', 'Min (An toàn)', 'Max (Dự phòng)']
                res['Tháng'] = res['Tháng'].dt.strftime('%m/%Y')
                st.dataframe(res.style.format('{:,.0f}', subset=['Trung bình', 'Min (An toàn)', 'Max (Dự phòng)']), use_container_width=True)
            else:
                st.warning(f"⚠️ Dữ liệu lịch sử của {label} quá ít để AI đưa ra dự báo.")

except Exception as e:
    st.error(f"Đã xảy ra lỗi hệ thống: {e}")
