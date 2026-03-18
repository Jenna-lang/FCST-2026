import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Dự báo Pareto & CIE 2026", layout="wide")
st.title("🚀 Hệ Thống Dự Báo Theo Khách Hàng & CIE 2026")

@st.cache_data
def load_data():
    df = pd.read_excel('AICheck.xlsx')
    df.columns = [str(col).strip() for col in df.columns]
    df['ds'] = pd.to_datetime(df['Requested deliv. date'])
    return df

try:
    df = load_data()
    
    # --- CẤU HÌNH SIDEBAR ---
    st.sidebar.header("⚙️ Cấu hình Dữ liệu")
    # Chọn cột Tên khách hàng (để tránh hiện mã số 700153)
    cust_col = st.sidebar.selectbox("Chọn cột Tên Khách Hàng:", df.columns)
    
    # Phân tích Pareto 85% doanh số
    summary = df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
    summary['Cum_Pct'] = summary['M USD'].cumsum() / summary['M USD'].sum()
    pareto_list = summary[summary['Cum_Pct'] <= 0.85]['Material name'].unique()

    selected_prod = st.sidebar.selectbox("1. Chọn mã linh kiện:", pareto_list)

    if selected_prod:
        prod_df = df[df['Material name'] == selected_prod].copy()
        tab1, tab2 = st.tabs(["📊 Phân bổ Khách hàng", "🎯 Dự báo Chi tiết & CIE"])

        with tab1:
            st.subheader(f"Cơ cấu khách hàng của mã: {selected_prod}")
            c1, c2 = st.columns(2)
            with c1:
                fig_p = px.pie(prod_df.groupby(cust_col)['M USD'].sum().reset_index(), 
                               values='M USD', names=cust_col, hole=0.4, title="Tỷ trọng doanh số")
                st.plotly_chart(fig_p, use_container_width=True)
            with c2:
                prod_df['Month'] = prod_df['ds'].dt.to_period('M').dt.to_timestamp()
                fig_b = px.bar(prod_df.groupby(['Month', cust_col])['Order qty.(A)'].sum().reset_index(), 
                               x='Month', y='Order qty.(A)', color=cust_col, title="Lịch sử đặt hàng theo tháng")
                st.plotly_chart(fig_b, use_container_width=True)

        with tab2:
            # --- LỰA CHỌN KHÁCH HÀNG CHI TIẾT ---
            cust_list = sorted([str(x) for x in prod_df[cust_col].unique()])
            options = ["TẤT CẢ (Bức tranh tổng quan)"] + cust_list
            selected_cust = st.selectbox("Chọn khách hàng để xem dự báo & chỉ số CIE:", options)
            
            if selected_cust == "TẤT CẢ (Bức tranh tổng quan)":
                working_df = prod_df.copy()
                display_name = "TỔNG THỂ THỊ TRƯỜNG"
            else:
                working_df = prod_df[prod_df[cust_col] == selected_cust].copy()
                display_name = selected_cust

            # Chuẩn bị dữ liệu cho Prophet
            working_df['ds_m'] = working_df['ds'].dt.to_period('M').dt.to_timestamp()
            p_df = working_df.groupby('ds_m')['Order qty.(A)'].sum().reset_index().rename(columns={'ds_m':'ds', 'Order qty.(A)':'y'})

            if len(p_df) >= 2:
                m = Prophet(yearly_seasonality=True).fit(p_df)
                future = m.make_future_dataframe(periods=12, freq='MS')
                fcst = m.predict(future)
                for c in ['yhat', 'yhat_lower', 'yhat_upper']: fcst[c] = fcst[c].clip(lower=0)
                
                st.divider()
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    st.subheader(f"📈 Dự báo FCST 2026: {display_name}")
                    st.pyplot(m.plot(fcst))
                
                with col_right:
                    st.subheader("💡 Chỉ số CIE & Lời khuyên")
                    f26 = fcst[fcst['ds'].dt.year == 2026]
                    avg26 = f26['yhat'].mean()
                    
                    # Hiển thị Metric
                    st.metric(f"Dự báo TB {display_name}", f"{avg26:,.0f} Pcs")
                    
                    # Giả lập chỉ số CIE dựa trên biến động dự báo (Volatility)
                    cie_score = 100 - ((f26['yhat_upper'].std() / avg26) * 100 if avg26 > 0 else 0)
                    st.write(f"**Chỉ số ổn định CIE:** `{cie_score:.1f}%`")
                    
                    if cie_score > 80:
                        st.success("✅ Khách hàng có độ tin cậy CIE cao. Duy trì mức tồn kho tiêu chuẩn.")
                    else:
                        st.warning("⚠️ CIE thấp (Biến động lớn). Cần tăng kho an toàn cho khách hàng này.")

                st.subheader(f"📋 Bảng chi tiết kế hoạch cung ứng cho {display_name}")
                res = f26[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                res.columns = ['Tháng', 'FCST Trung bình', 'CIE Min (Safety)', 'CIE Max (Buffer)']
                res['Tháng'] = res['Tháng'].dt.strftime('%m/%Y')
                st.dataframe(res.style.format('{:,.0f}', subset=['FCST Trung bình', 'CIE Min (Safety)', 'CIE Max (Buffer)']), use_container_width=True)
            else:
                st.warning(f"⚠️ Dữ liệu của {display_name} chưa đủ để thực hiện dự báo chi tiết.")

except Exception as e:
    st.error(f"Lỗi thực thi: {e}")
