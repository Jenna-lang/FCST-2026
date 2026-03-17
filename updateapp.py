import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dự báo theo Khách hàng 2026", layout="wide")
st.title("📊 Quản Trị Cung Ứng: Chi Tiết Theo Khách Hàng")

@st.cache_data
def load_data():
    # Đọc file dữ liệu từ GitHub
    df = pd.read_excel('AICheck.xlsx')
    df['ds'] = pd.to_datetime(df['Requested deliv. date'])
    return df

try:
    df = load_data()

    # 1. BỘ LỌC TẠI SIDEBAR
    st.sidebar.header("Bộ lọc truy vấn")
    
    # Lọc linh kiện (Material)
    all_materials = sorted(df['Material name'].unique())
    selected_prod = st.sidebar.selectbox("1. Chọn mã linh kiện:", all_materials)

    # Lọc khách hàng (Customer) dựa trên linh kiện đã chọn
    customers_of_prod = sorted(df[df['Material name'] == selected_prod]['End Cust'].unique())
    selected_cust = st.sidebar.selectbox("2. Chọn khách hàng:", customers_of_prod)

    if selected_prod and selected_cust:
        # Lọc dữ liệu kết hợp cả 2 điều kiện
        mask = (df['Material name'] == selected_prod) & (df['End Cust'] == selected_cust)
        data = df[mask].copy()
        
        # Gom nhóm theo tháng
        data['ds_month'] = data['ds'].dt.to_period('M').dt.to_timestamp()
        p_df = data.groupby('ds_month')['Order qty.(A)'].sum().reset_index().rename(columns={'ds_month':'ds', 'Order qty.(A)':'y'})

        st.divider()
        st.header(f"📍 Khách hàng: {selected_cust}")
        st.caption(f"Sản phẩm: {selected_prod}")

        if len(p_df) >= 2:
            # Chạy AI Prophet
            m = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
            m.fit(p_df)
            future = m.make_future_dataframe(periods=12, freq='MS')
            forecast = m.predict(future)
            for col in ['yhat', 'yhat_lower', 'yhat_upper']: 
                forecast[col] = forecast[col].clip(lower=0)

            # HIỂN THỊ KẾT QUẢ
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("📈 Biểu đồ dự báo nhu cầu")
                fig = m.plot(forecast)
                st.pyplot(fig)
            
            with col2:
                st.subheader("💡 Lời khuyên cho Sales")
                f_2026 = forecast[forecast['ds'].dt.year == 2026]
                avg_demand = f_2026['yhat'].mean()
                st.metric("Nhu cầu TB tháng (2026)", f"{avg_demand:,.0f} Pcs")
                
                # Cảnh báo dựa trên lịch sử khách hàng
                if p_df['y'].std() > p_df['y'].mean():
                    st.error("🚩 Khách hàng này đặt hàng không đều. Hãy bám sát mức 'Max' để tránh đứt hàng.")
                else:
                    st.success("✅ Khách hàng đặt hàng ổn định. Có thể tối ưu tồn kho theo mức 'Trung bình'.")

            # BẢNG DỮ LIỆU CHI TIẾT 2026
            st.subheader(f"📊 Kế hoạch cung ứng 2026 cho {selected_cust}")
            res_2026 = f_2026[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            res_2026.columns = ['Tháng', 'Dự báo TB', 'Min (An toàn)', 'Max (Kỳ vọng)']
            res_2026['Tháng'] = res_2026['Tháng'].dt.strftime('%m/%Y')
            st.dataframe(res_2026.style.format('{:,.0f}', subset=['Dự báo TB', 'Min (An toàn)', 'Max (Kỳ vọng)']), use_container_width=True)
            
        else:
            st.warning(f"⚠️ Khách hàng {selected_cust} có quá ít lịch sử đặt mã hàng này (dưới 2 tháng) để AI dự báo.")

except Exception as e:
    st.error(f"Lỗi: {e}. Vui lòng kiểm tra file AICheck.xlsx")
