import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

st.set_page_config(page_title="Dự báo Sản xuất theo Màu CIE", layout="wide")
st.title("🎨 Hệ Thống Dự Báo Sản Lượng & Chỉ Số Màu CIE 2026")

@st.cache_data
def load_data():
    df = pd.read_excel('AICheck.xlsx')
    df.columns = [str(col).strip() for col in df.columns]
    df['ds'] = pd.to_datetime(df['Requested deliv. date'])
    return df

try:
    df = load_data()
    
    # --- CẤU HÌNH SIDEBAR ---
    st.sidebar.header("⚙️ Thiết lập dữ liệu")
    # Tự động nhận diện cột Tên khách hàng và cột CIE (màu sắc)
    cust_col = st.sidebar.selectbox("Chọn cột Tên Khách Hàng:", df.columns, index=0)
    cie_col = st.sidebar.selectbox("Chọn cột Chỉ số màu CIE:", df.columns)

    # Lọc danh sách sản phẩm Pareto
    summary = df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
    pareto_list = summary.head(20)['Material name'].unique()

    selected_prod = st.sidebar.selectbox("1. Chọn Sản phẩm cần dự báo:", pareto_list)

    if selected_prod:
        prod_df = df[df['Material name'] == selected_prod].copy()
        
        st.subheader(f"📊 Phân tích mã hàng: {selected_prod}")
        
        # Lựa chọn Khách hàng
        cust_list = sorted([str(x) for x in prod_df[cust_col].unique()])
        selected_cust = st.selectbox("2. Chọn Khách hàng:", ["TẤT CẢ"] + cust_list)

        if selected_cust == "TẤT CẢ":
            working_df = prod_df.copy()
        else:
            working_df = prod_df[prod_df[cust_col] == selected_cust].copy()

        # Hiển thị bảng màu CIE hiện có của sản phẩm này
        cie_options = sorted([str(x) for x in working_df[cie_col].unique()])
        selected_cie = st.multiselect("3. Lọc theo chỉ số màu CIE (Có thể chọn nhiều):", cie_options, default=cie_options)

        final_df = working_df[working_df[cie_col].astype(str).isin(selected_cie)].copy()

        # --- XỬ LÝ DỰ BÁO AI ---
        final_df['ds_m'] = final_df['ds'].dt.to_period('M').dt.to_timestamp()
        p_df = final_df.groupby('ds_m')['Order qty.(A)'].sum().reset_index().rename(columns={'ds_m':'ds', 'Order qty.(A)':'y'})

        if len(p_df) >= 2:
            m = Prophet(yearly_seasonality=True).fit(p_df)
            future = m.make_future_dataframe(periods=12, freq='MS')
            fcst = m.predict(future)
            f26 = fcst[fcst['ds'].dt.year == 2026].copy()
            
            # Chặn âm
            f26['yhat'] = f26['yhat'].clip(lower=0)

            col_chart, col_data = st.columns([1, 1])
            
            with col_chart:
                st.write(f"**Xu hướng nhu cầu 2026 ({selected_prod})**")
                st.pyplot(m.plot(fcst))
            
            with col_data:
                st.write("**Bảng dự báo cụ thể theo Sản phẩm & Màu CIE**")
                # Tạo bảng tổng hợp cuối cùng
                report = f26[['ds', 'yhat']].copy()
                report.columns = ['Tháng', 'Số lượng dự báo (Pcs)']
                report['Tên Sản phẩm'] = selected_prod
                report['Chỉ số màu CIE'] = ", ".join(selected_cie) if len(selected_cie) < 4 else "Nhiều mã màu"
                report['Khách hàng'] = selected_cust
                
                # Sắp xếp lại cột cho đúng ý bạn
                report = report[['Tháng', 'Tên Sản phẩm', 'Chỉ số màu CIE', 'Khách hàng', 'Số lượng dự báo (Pcs)']]
                report['Tháng'] = report['Tháng'].dt.strftime('%m/%Y')
                
                st.dataframe(report.style.format('{:,.0f}', subset=['Số lượng dự báo (Pcs)']), use_container_width=True)
                
                # Lời khuyên sản xuất
                total_2026 = report['Số lượng dự báo (Pcs)'].sum()
                st.info(f"💡 **Tổng nhu cầu 2026:** {total_2026:,.0f} Pcs cho các mã màu CIE đã chọn.")
        else:
            st.warning("⚠️ Dữ liệu lịch sử quá ít để lập bảng dự báo theo màu sắc này.")

except Exception as e:
    st.error(f"Lỗi: {e}. Vui lòng kiểm tra lại tên cột trong file Excel.")
