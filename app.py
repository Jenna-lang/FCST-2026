import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pareto Forecast 2026", layout="wide")
st.title("📊 Hệ Thống Dự Báo Pareto 2026")

@st.cache_data
def load_data():
    df = pd.read_excel('AICheck.xlsx')
    df['ds'] = pd.to_datetime(df['Requested deliv. date'])
    return df

df = load_data()

# Lọc danh sách sản phẩm Pareto (85% giá trị)
summary = df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
summary['Cum_Pct'] = summary['M USD'].cumsum() / summary['M USD'].sum()
pareto_list = summary[summary['Cum_Pct'] <= 0.85]['Material name'].unique()

selected_prod = st.sidebar.selectbox("Chọn mã hàng Pareto:", pareto_list)

if selected_prod:
    data = df[df['Material name'] == selected_prod].copy()
    data['ds_month'] = data['ds'].dt.to_period('M').dt.to_timestamp()
    p_df = data.groupby('ds_month')['Order qty.(A)'].sum().reset_index().rename(columns={'ds_month':'ds', 'Order qty.(A)':'y'})

    if len(p_df) >= 2:
        # AI Forecast
        m = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
        m.fit(p_df)
        future = m.make_future_dataframe(periods=12, freq='MS')
        forecast = m.predict(future)
        for col in ['yhat', 'yhat_lower', 'yhat_upper']: forecast[col] = forecast[col].clip(lower=0)

        # 1. Biểu đồ
        st.subheader(f"📈 Xu hướng dự báo: {selected_prod}")
        fig = m.plot(forecast)
        st.pyplot(fig)

        # 2. Lời khuyên AI
        st.subheader("💡 Chiến lược tồn kho")
        f_2026 = forecast[forecast['ds'].dt.year == 2026]
        trend = ((f_2026['yhat'].iloc[-1] - f_2026['yhat'].iloc[0]) / f_2026['yhat'].iloc[0] * 100) if f_2026['yhat'].iloc[0] > 0 else 0
        if trend > 10:
            st.success(f"Dự báo tăng mạnh {trend:.1f}%. Cần đặt thêm hàng dự phòng.")
        else:
            st.info("Nhu cầu ổn định. Duy trì Safety Stock.")

        # 3. Bảng số liệu 2026 (Fix lỗi không hiển thị)
        st.subheader("📋 Chi tiết kế hoạch đặt hàng 2026")
        res_2026 = f_2026[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        res_2026.columns = ['Tháng', 'Trung bình', 'Min (An toàn)', 'Max (Kỳ vọng)']
        res_2026['Tháng'] = res_2026['Tháng'].dt.strftime('%m/%Y')
        st.table(res_2026.style.format('{:,.0f}', subset=['Trung bình', 'Min (An toàn)', 'Max (Kỳ vọng)']))
    else:
        st.warning("⚠️ Dữ liệu lịch sử của mã hàng này quá ít để AI phân tích.")
