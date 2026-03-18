import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

st.set_page_config(page_title="Dự báo chi tiết màu CIE", layout="wide")
st.title("🎨 Dự báo Sản lượng chi tiết theo từng mã màu CIE 2026")

@st.cache_data
def load_data():
    df = pd.read_excel('AICheck.xlsx')
    df.columns = [str(col).strip() for col in df.columns]
    df['ds'] = pd.to_datetime(df['Requested deliv. date'])
    return df

try:
    df = load_data()
    
    st.sidebar.header("⚙️ Thiết lập")
    cust_col = st.sidebar.selectbox("Cột Khách hàng:", df.columns)
    cie_col = st.sidebar.selectbox("Cột Mã màu CIE:", df.columns)
    
    # Lọc sản phẩm
    prod_list = df['Material name'].unique()
    selected_prod = st.sidebar.selectbox("Chọn Sản phẩm:", prod_list)

    if selected_prod:
        prod_df = df[df['Material name'] == selected_prod].copy()
        
        # Lọc Khách hàng
        cust_list = sorted(prod_df[cust_col].unique())
        selected_cust = st.selectbox("Chọn Khách hàng:", ["ALL"] + cust_list)
        
        working_df = prod_df if selected_cust == "ALL" else prod_df[prod_df[cust_col] == selected_cust]
        
        # Chọn mã màu CIE cần dự báo riêng
        cie_options = sorted(working_df[cie_col].unique().astype(str))
        selected_cies = st.multiselect("Chọn các mã màu CIE cần xem dự báo riêng:", cie_options, default=cie_options[:3])

        if selected_cies:
            all_forecasts = []
            
            for cie in selected_cies:
                # Lọc dữ liệu riêng cho từng mã màu
                cie_df = working_df[working_df[cie_col].astype(str) == cie].copy()
                cie_df['ds_m'] = cie_df['ds'].dt.to_period('M').dt.to_timestamp()
                p_df = cie_df.groupby('ds_m')['Order qty.(A)'].sum().reset_index().rename(columns={'ds_m':'ds', 'Order qty.(A)':'y'})
                
                if len(p_df) >= 2:
                    m = Prophet(yearly_seasonality=True).fit(p_df)
                    future = m.make_future_dataframe(periods=12, freq='MS')
                    fcst = m.predict(future)
                    
                    # Lấy kết quả 2026
                    f26 = fcst[fcst['ds'].dt.year == 2026][['ds', 'yhat']].copy()
                    f26['yhat'] = f26['yhat'].clip(lower=0)
                    f26['Mã màu CIE'] = cie
                    f26['Tên sản phẩm'] = selected_prod
                    all_forecasts.append(f26)
            
            if all_forecasts:
                final_report = pd.concat(all_forecasts)
                final_report['Tháng'] = final_report['ds'].dt.strftime('%m/%Y')
                final_report = final_report.rename(columns={'yhat': 'Số lượng dự báo (Pcs)'})
                
                # --- HIỂN THỊ KẾT QUẢ ---
                st.subheader("📋 Bảng dự báo chi tiết theo từng mã màu")
                
                # Biểu đồ so sánh các mã màu
                fig = px.line(final_report, x='Tháng', y='Số lượng dự báo (Pcs)', color='Mã màu CIE',
                              title=f"So sánh dự báo các mã màu CIE của {selected_prod}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Bảng số liệu xoay (Pivot table) để dễ nhìn
                pivot_df = final_report.pivot(index='Tháng', columns='Mã màu CIE', values='Số lượng dự báo (Pcs)').fillna(0)
                st.write("**Bảng số liệu tổng hợp (Pcs):**")
                st.dataframe(pivot_df.style.format("{:,.0f}"), use_container_width=True)
                
                # Bảng chi tiết đầy đủ
                with st.expander("Xem bảng chi tiết đầy đủ"):
                    st.dataframe(final_report[['Tháng', 'Tên sản phẩm', 'Mã
